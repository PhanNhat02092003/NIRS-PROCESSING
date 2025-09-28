import os
import joblib
from model.classification_model import *
import numpy as np
import json

def load_vegetable_classification_model(pretrained_folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SMARTNIR(cfg=SmartNIRClassificationConfig(
        signal_len=2136,
        out_ch_per_branch=64,
        d_model=128,
        depth=3,
        n_heads=4,
        classifier="kan",   # đổi "mlp" nếu muốn MLP
        num_classes=9
    )).to(device)

    model.load_state_dict(torch.load(os.path.join(pretrained_folder, "checkpoint.pth"), map_location=device))
    model.to(device)
    model.eval()

    stats = np.load(os.path.join(pretrained_folder, "stats.npz"))
    mean, std = stats["mean"], stats["std"] 

    with open(os.path.join(pretrained_folder, "idx2label.json"), "r", encoding="utf-8") as f:
        label2idx = json.load(f)

    idx2label = {v: k for k, v in label2idx.items()}
    
    return model, mean, std, idx2label

def load_verify_substances_models(pretrained_folder):
    model_paths = {
        "Thiamethoxam": None,
        "Permethrin": None,
        "Metalaxyl": None,
        "Azoxystrobin": None,
        "Imidaclopird": None,
        "Difenoconazole": None,
        "Cypermethrin": None,
        "Cyhalothrin": None,
        "Chlorantraniliprol": None,
        "Chlopyrifos Methyl": None,
        "Emamectin benzoat": None,
        "Chlorothalonil": None,
        "Chlothinidin": None,
        "Triadimefon": None,
        "Triadimefon": None, 
        "Cyantraniliprole": None,
        "Flutolanil": None,
        "Indoxacarb": None,
        "Abamectin": None,
        "Propamocarb.HCL": None,
        "Chlothianidin": None,
        "Emamectin benzoate": None
    }
    for path in os.listdir(pretrained_folder):
        substance = path.split("_")[-1].replace(".pkl", "")
        model_paths[substance] = os.path.join(pretrained_folder, path)

    pipelines = {}

    for substance in model_paths.keys():
        if model_paths[substance]:
            if not os.path.exists(model_paths[substance]):
                pipelines[substance] = None

            pipelines[substance] = joblib.load(model_paths[substance])
        else:
            pipelines[substance] = None
    
    return pipelines

def load_predict_substances_concentration_models(pretrained_folder):
    model_paths = {
        "Thiamethoxam": None,
        "Permethrin": None,
        "Metalaxyl": None,
        "Azoxystrobin": None,
        "Imidaclopird": None,
        "Difenoconazole": None,
        "Cypermethrin": None,
        "Cyhalothrin": None,
        "Chlorantraniliprol": None,
        "Chlopyrifos Methyl": None,
        "Emamectin benzoat": None,
        "Chlorothalonil": None,
        "Chlothinidin": None,
        "Triadimefon": None,
        "Triadimefon": None, 
        "Cyantraniliprole": None,
        "Flutolanil": None,
        "Indoxacarb": None,
        "Abamectin": None,
        "Propamocarb.HCL": None,
        "Chlothianidin": None,
        "Emamectin benzoate": None
    }
    for path in os.listdir(pretrained_folder):
        substance = path.split("_")[-1].replace(".pkl", "")
        model_paths[substance] = os.path.join(pretrained_folder, path)

    pipelines = {}

    for substance in model_paths.keys():
        if model_paths[substance]:
            if not os.path.exists(model_paths[substance]):
                pipelines[substance] = None

            pipelines[substance] = joblib.load(model_paths[substance])
        else:
            pipelines[substance] = None
    
    return pipelines

