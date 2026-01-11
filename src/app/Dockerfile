# ---------- Base image ---------- #
FROM python:3.10-slim

# ---------- Metadata ---------- #
LABEL maintainer='IuliiaK'
LABEL description='FastAPI microservice with Prometheus monitoring'

# ---------- Environment ---------- #
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------- Workdir ---------- #
WORKDIR /app

# ---------- Copy files ---------- #
COPY requirements.txt /app/requirements.txt
COPY services/ml_service /app/ml_service
COPY services/models /app/models

# ---------- Install dependencies ---------- #
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---------- Expose port ---------- #
EXPOSE 8080

# ---------- Start FastAPI app ---------- #
CMD ["uvicorn", "ml_service.main_stage_3_4:app", "--host", "0.0.0.0", "--port", "8080"]
