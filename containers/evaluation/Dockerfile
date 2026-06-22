FROM python:3.12-slim

WORKDIR /app

COPY containers/evaluation/requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY containers/evaluation/main.py .
COPY tsf2_core /app/tsf2_core
ENV PYTHONPATH=/app

CMD ["python", "main.py"]
