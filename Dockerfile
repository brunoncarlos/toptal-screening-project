FROM python:3.12-slim

WORKDIR /app

# PyTorch CPU compatible
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --default-timeout=200 --no-cache-dir -r requirements.txt

# COPY YOUR PROJECT FILES
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY predictions_code.py .

CMD ["python", "predictions_code.py"]