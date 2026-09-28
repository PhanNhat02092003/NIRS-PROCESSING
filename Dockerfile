FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.deploy.txt .
RUN pip install --no-cache-dir -r requirements.deploy.txt

COPY . .

EXPOSE 9000

CMD ["python3", "app.py"]
