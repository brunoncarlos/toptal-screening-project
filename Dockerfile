FROM python:3.12-slim

WORKDIR /app

# Instalar PyTorch CPU (evita CUDA)
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY predictions_code.py .

CMD ["python", "predictions_code.py"]
