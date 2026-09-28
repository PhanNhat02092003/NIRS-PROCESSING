"""EBAR: Error Based Accumulation Regression, reimplemented from the paper
(baseline-information/regression/EBAR/J.UCS Article Template V5.tex) as a
classical-ML stacking baseline for Stage 2 (concentration prediction),
compared against SMART-NIR.

9 base learners (SVR x3 kernels, LASSO, RIDGE, Elastic Net, PLS, Random
Forest, XGBoost) feed out-of-fold predictions into a LightGBM meta-model
(paper's Algorithm 1/2) -- implemented via sklearn's StackingRegressor,
which performs exactly this out-of-fold-then-refit-on-full-data scheme.

Three deliberate deviations from the paper:
  - The paper's dataset was 332 samples total, so exact-kernel SVR (SVR's
    fit cost is superlinear, effectively O(n^2)-O(n^3)) was never a
    bottleneck. This project's substances have thousands to ~25k samples
    per fold, where SVR-poly/rbf would take minutes to hours per fit.
    `SubsampledRegressor` below fits SVR on a capped random subsample
    (default 3000, empirically ~1s to fit) instead of the full fold.
  - The paper tunes every model's hyperparameters via GridSearch on its own
    (very different, tiny) dataset; those tuned values aren't meaningful
    here. This uses the *CV variants (LassoCV/RidgeCV/ElasticNetCV) for
    cheap built-in regularization-strength selection, and reasonable fixed
    defaults for the tree ensembles, instead of re-running GridSearch per
    substance/fold/machine.
  - The paper's target preprocessing is plain z-score; this project's
    substances (e.g. Permethrin: median ~1.02, 90th percentile ~216) are far
    more right-skewed than the paper's dataset, so the engine log1p's the
    target first (see regression.py::fit_xy_scaler) -- without
    it, MSE-driven fitting is graded almost entirely on the thin
    high-concentration tail and base learners undershoot it badly (observed
    R2 as low as -2.46 on Permethrin, worse than predicting the mean).
"""
from typing import Optional

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.svm import SVR
from xgboost import XGBRegressor


# Both wrapper classes below inherit (RegressorMixin, BaseEstimator) in that
# order deliberately, not (BaseEstimator, RegressorMixin): on sklearn>=1.6's
# tag system, RegressorMixin.__sklearn_tags__() calls super().__sklearn_tags__()
# to set estimator_type="regressor", which only takes effect if it's earlier
# in the MRO than BaseEstimator's own __sklearn_tags__(). Getting this
# backwards still leaves `_estimator_type == "regressor"` (the old attribute)
# looking correct, but sklearn.base.is_regressor() -- what StackingRegressor
# actually checks -- silently returns False, raising "should be a regressor".
class SubsampledRegressor(RegressorMixin, BaseEstimator):
    """Wraps a regressor so `.fit` trains on at most `max_samples` randomly
    chosen rows -- keeps SVR (poly/rbf kernels) tractable on large folds
    without changing `.predict` behavior (predict always uses the full
    input, since only fitting scales badly with sample count).
    """

    def __init__(self, estimator, max_samples: int = 3000, random_state: int = 42):
        self.estimator = estimator
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y):
        n = X.shape[0]
        if n > self.max_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(n, size=self.max_samples, replace=False)
            X, y = X[idx], y[idx]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class ClippedRegressor(RegressorMixin, BaseEstimator):
    """Wraps a regressor, clipping predictions to `clip_range` -- a safety
    net against numerically pathological base learners feeding garbage into
    the stacking meta-model. Concretely observed on Permethrin (FLAMENIR):
    SVR-linear/poly don't fully converge within `max_iter` on highly
    collinear NIR wavelengths, and RidgeCV's default alpha grid is too weak
    for that collinearity ("Ill-conditioned matrix" warnings) -- both
    produced predictions many orders of magnitude outside any sane range
    (svr_poly's pred_max had ~280 digits; ridge's R2 was -56163), which
    corrupted the LightGBM meta-model's out-of-fold training features. Every
    base learner here is trained on a log1p+z-scored target with std=1; the
    true observed range for these substances is typically within roughly
    [-1, +2.5] std (e.g. Permethrin spans [-0.88, +2.28]), so +-4 std is
    already generous headroom for real large deviations while still
    catching genuine blow-ups -- a looser bound like +-10 std still lets
    expm1() (the target's inverse transform) turn a saturated prediction
    into a nonsensical hundreds-of-millions concentration.
    """

    def __init__(self, estimator, clip_range=(-4.0, 4.0)):
        self.estimator = estimator
        self.clip_range = clip_range

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        pred = self.estimator_.predict(X)
        return np.clip(pred, self.clip_range[0], self.clip_range[1])


def _pls_components(n_samples: int, n_features: int, target: int = 10) -> int:
    """PLSRegression requires n_components <= min(n_samples, n_features)."""
    return max(1, min(target, n_samples - 1, n_features))


def build_ebar_model(n_samples: int, n_features: int, max_svr_samples: int = 3000,
                      inner_cv: int = 5, random_state: int = 42,
                      n_jobs: Optional[int] = 1) -> StackingRegressor:
    """Builds the EBAR stacking ensemble: 9 base learners + LightGBM
    meta-model, combined via sklearn's StackingRegressor (which implements
    the paper's out-of-fold-stacking Algorithm 1/2: `cv`-fold out-of-fold
    predictions train the meta-model; each base learner is also refit on
    the full training fold for use at predict time).

    Parallelism is pushed down into RandomForest/XGBoost themselves
    (n_jobs=-1, since tree ensembles on ~10-25k samples are the real
    bottleneck and parallelize very well) rather than across the
    StackingRegressor's 9 base estimators (kept at n_jobs=1): the linear/SVR
    models are cheap regardless, so parallelizing *those* while serializing
    the two genuinely expensive models wastes the CPU budget where it
    matters. Running both levels at n_jobs=-1 simultaneously oversubscribes
    the CPU (72 outer jobs x 72 inner threads) and is far slower than either
    single-level scheme.
    """
    svr_kwargs = dict(max_samples=max_svr_samples, random_state=random_state)
    # sklearn's SVR defaults to max_iter=-1 (unlimited) -- its libsvm/SMO
    # solver can take a very long time to converge on highly collinear NIR
    # wavelengths, occasionally appearing to hang. Cap iterations so a fit
    # always terminates in bounded time; an early-stopped (not fully
    # converged) solution is an acceptable tradeoff over a fit that never
    # returns. 20000 wasn't tight enough: on Chlorantraniliprol/OCEANFX
    # (only 2310 train rows -- *below* the 3000-row subsample cap, so this
    # isn't a large-data issue) svr_linear/svr_poly each measured ~350s for
    # a single fit; the outer StackingRegressor calls each base learner 6x
    # (5 inner-CV folds + 1 refit), so that's ~70 min for two learners that
    # individually aren't even the ensemble's strongest (R2 -1.39 / 0.22).
    # Some substances' data is just harder for libsvm to converge on,
    # independent of row count. 5000 bounds the worst case much tighter.
    svr_max_iter = 5000
    base_learners = [
        ("svr_linear", SubsampledRegressor(SVR(kernel="linear", max_iter=svr_max_iter), **svr_kwargs)),
        ("svr_poly", SubsampledRegressor(SVR(kernel="poly", degree=3, max_iter=svr_max_iter), **svr_kwargs)),
        ("svr_rbf", SubsampledRegressor(SVR(kernel="rbf", max_iter=svr_max_iter), **svr_kwargs)),
        # NIR wavelengths are highly collinear (adjacent points near-identical),
        # which makes coordinate descent converge slowly, especially at the
        # near-zero-regularization end of the default alpha path. Raising
        # max_iter alone doesn't fully fix it, so also shrink the alpha grid
        # (fewer, less extreme candidates -> faster and better-behaved) via
        # n_alphas/eps. RidgeCV's own default alpha grid -- (0.1, 1.0, 10.0)
        # -- is far too weakly regularized for this collinearity: observed
        # producing "Ill-conditioned matrix" warnings and predictions blown
        # up to ~1.4e6 on held-out data, so it's given a wider/stronger grid.
        # cv=None (not cv=5) is deliberate too: with an explicit K-fold cv,
        # RidgeCV manually refits plain Ridge() cv x n_alphas times per call
        # (x 6 calls from the outer StackingRegressor), and on this data one
        # of those fits alone measured 360s (vs ~9s for the whole alpha
        # search below) -- some (alpha, fold) combination hits a much slower
        # numerical path on the "Ill-conditioned matrix" warning. cv=None
        # instead uses sklearn's efficient closed-form generalized
        # cross-validation (one SVD decomposition scores every alpha at
        # once), which is both ~40x faster here and scored a healthier R2
        # (0.49 vs negative) in isolated testing on Thiamethoxam/OCEANFX.
        # n_jobs=-1 here (not 1): the outer StackingRegressor itself uses
        # n_jobs=1, meaning base learners are fit one at a time -- so unlike
        # RandomForest/XGBoost (which also get n_jobs=-1), LassoCV/ElasticNetCV
        # never actually run *concurrently* with anything else that wants
        # every core, so there's no real oversubscription risk in parallelizing
        # their own 5-fold x 30-alpha inner search. Measured 5.3x faster
        # (40.4s -> 7.6s) on Thiamethoxam/OCEANFX with identical R2.
        # max_iter=5000 (not 20000, same reasoning as svr_max_iter above):
        # on Chlorothalonil/OCEANFX (target has only 6 distinct concentration
        # values, unevenly spaced -- 0.132/0.27/4.3/5.6/7.9/12.3) lasso alone
        # measured 333.6s and enet 204.7s for one fit; coordinate descent
        # struggles to converge on some (alpha, fold) combination against
        # that near-discrete target, independent of n_jobs. An early-stopped
        # solution here is an acceptable tradeoff, same as the SVR cap.
        ("lasso", LassoCV(cv=5, random_state=random_state, max_iter=5000, n_alphas=30, eps=1e-2, n_jobs=-1)),
        ("ridge", RidgeCV(alphas=np.logspace(-1, 6, 12), cv=None, gcv_mode="svd")),
        ("enet", ElasticNetCV(cv=5, random_state=random_state, max_iter=5000, n_alphas=30, eps=1e-2, n_jobs=-1)),
        ("pls", PLSRegression(n_components=_pls_components(n_samples, n_features))),
        # max_features="sqrt" (not sklearn's default of 1.0 = all features):
        # with 2136 wavelength features, considering every one of them at
        # every split of every one of 200 trees was, by far, this ensemble's
        # single biggest cost -- 237.6s for one fit on Thiamethoxam/OCEANFX
        # (6 calls from the outer StackingRegressor puts it at ~24 min alone).
        # sqrt(2136)~=46 features/split is the standard random-forest
        # heuristic anyway; measured 48x faster (4.9s) with slightly *better*
        # held-out R2 (0.9292 vs 0.9246), so this is a pure win, not a
        # speed/quality tradeoff.
        ("random_forest", RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                                  random_state=random_state, n_jobs=-1)),
        ("xgboost", XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  random_state=random_state, n_jobs=-1, verbosity=0)),
    ]
    base_learners = [(name, ClippedRegressor(est)) for name, est in base_learners]
    # The meta-model's 9 input features are a mix of a few genuinely strong
    # learners (random_forest/xgboost) and several weak/unstable ones
    # (svr_linear/svr_poly/svr_rbf/ridge -- individually near or below R2=0
    # on this data). An unregularized LGBMRegressor happily overfits to
    # whatever fold-specific noise those weak out-of-fold features carry:
    # observed on Permethrin with feature_importances_ giving svr_rbf/ridge
    # *more* weight than random_forest/xgboost, and the resulting stacked
    # R2 (-0.18) coming out *worse* than a plain average of just RF+XGB's
    # own predictions (+0.44). Shallow trees + fewer leaves + a stronger
    # min_child_samples/reg_alpha/reg_lambda keep the meta-model from
    # chasing that noise.
    meta_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3,
                                num_leaves=7, min_child_samples=100,
                                reg_alpha=1.0, reg_lambda=1.0,
                                random_state=random_state, verbosity=-1)

    return StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_model,
        cv=inner_cv,
        n_jobs=n_jobs,
        passthrough=False,
    )
