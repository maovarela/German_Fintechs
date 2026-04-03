FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir "kagglehub[pandas-datasets]"

COPY German_Fintech.py .

CMD ["python", "German_Fintech.py"]
