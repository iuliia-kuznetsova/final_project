# Bank Products Recommender API

A FastAPI-based recommendation service for bank products with Prometheus metrics and Grafana dashboards.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose Stack                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Recommender   │  │   Prometheus    │  │   Grafana    │ │
│  │      API        │──│   (Metrics)     │──│ (Dashboard)  │ │
│  │    :8080        │  │    :9090        │  │    :3000     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────┐                                         │
│  │   OvR Model     │                                         │
│  │   (CatBoost)    │                                         │
│  └─────────────────┘                                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Build and Run with Docker Compose

```bash
# Build and start all services
docker compose -f src/api/docker-compose.yml up -d --build

# Check service status
docker compose ps

# View logs
docker compose logs -f recommender
```

rebuild containers
# After code changes, run:
docker compose -f src/api/docker-compose.yml build --no-cache recommender
docker compose -f src/api/docker-compose.yml up -d

After rebuilding, you may need to fix logs permissions again:
sudo chown -R 1000:1000 /home/mle-user/mle_projects/final_project/logs
sudo chown -R mle-user:mle-user /home/mle-user/mle_projects/final_project/logs/


### 3. Access Services

- **Recommender API**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)

## API Endpoints
/           - API info
/health     - Health check
/metrics    - Prometheus metrics
/model/info - Model information
/predict    - Single prediction


### Health Check
```curl 
-s http://localhost:8080/health | jq
```

```bash
GET /health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### API Root Info
```curl 
-s http://localhost:8080/ | jq
```

Expected response:
```json
{
  "name": "Bank Products Recommender API",
  "status": "running",
  "docs": "/docs",
  "health": "/health",
  "metrics": "/metrics"
}
```

### Model Info
```curl 
-s http://localhost:8080/model/info | jq
```

Expected response:
```json
{
  "model_name": "ovr_grouped_catboost",
  "model_version": "20260116",
  "n_features": 27,
  "n_cat_features": 33,
  "n_products": 24,
  "groups": {
    "frequent": 8,
    "mid": 7,
    "rare": 9
  },
  "top_k_default": 7,
  "is_loaded": true
}
```

### Single Prediction
```curl 
-s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "12345678",
    "features": {
      "age": 35,
      "renta": 50000.0,
      "customer_period": 12,
      "ind_nuevo": false,
      "indresi": true,
      "indfall": false,
      "ind_actividad_cliente": true,
      "ind_empleado": "A",
      "pais_residencia": "ES",
      "sexo": "H",
      "indrel": "1",
      "indrel_1mes": "1",
      "tiprel_1mes": "A",
      "canal_entrada": "KHE",
      "cod_prov": "28",
      "segmento": "02 - PARTICULARES"
    },
    "top_k": 7
  }' | jq
```

```bash
POST /predict
```

Request:
```json
{
  "customer_id": "12345678",
  "features": {
    "fecha_dato": "2016-05-28",
    "ncodpers": "12345678",
    "age": 35,
    "customer_period": 12,
    "ind_nuevo": false,
    "indresi": true,
    "indfall": false,
    "ind_actividad_cliente": true,
    "ind_cco_fin_ult1": true
  },
  "top_k": 7
}
```

Response:
```json
{
  "customer_id": "12345678",
  "recommendations": [
    {
      "product_id": "target_recibo",
      "product_name": "Direct Debit",
      "probability": 0.85,
      "rank": 1
    },
    {
      "product_id": "target_nomina",
      "product_name": "Payroll",
      "probability": 0.72,
      "rank": 2
    }
  ],
  "latency_ms": 45.2,
  "model_version": "20240115"
}
```

### Batch Prediction

```bash
POST /predict/batch
```

Request:
```json
{
  "customers": [
    {
      "customer_id": "12345678",
      "features": {...},
      "top_k": 7
    },
    {
      "customer_id": "87654321",
      "features": {...},
      "top_k": 7
    }
  ]
}
```

### Model Info

```bash
GET /model/info
```

### Metrics
curl -s http://localhost:8080/metrics | head -50

## Testing the API

### Run API Tests

```bash
# Generate sample data and run tests
python -m src.api.test_api --sample --limit 100 --sleep 0.1

# Test with custom configuration
python -m src.api.test_api \
  --host localhost \
  --port 8080 \
  --limit 50 \
  --batch-size 10 \
  --top-k 7
```

### Example with curl

```bash
# Health check
curl http://localhost:8080/health

# Single prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "12345678",
    "features": {
      "fecha_dato": "2016-05-28",
      "ncodpers": "12345678",
      "ind_empleado": "A",
      "pais_residencia": "ES",
      "sexo": "H",
      "age": 35,
      "ind_nuevo": false,
      "indrel": "1",
      "indrel_1mes": "1",
      "tiprel_1mes": "A",
      "indresi": true,
      "canal_entrada": "KAT",
      "indfall": false,
      "cod_prov": "28",
      "ind_actividad_cliente": true,
      "renta": 326124.90,
      "segmento": "02 - PARTICULARES",
      "customer_period": 12
    },
    "top_k": 7
  }'
```

Expected 
{"customer_id":"12345678","recommendations":[{"product_id":"target_recibo","product_name":"Direct Debit","probability":0.0,"rank":1},{"product_id":"target_cno_fin","product_name":"Payroll Account","probability":0.0,"rank":2},{"product_id":"target_cco_fin","product_name":"Current Account","probability":0.0,"rank":3},{"product_id":"target_ctma_fin","product_name":"Más Particular Account","probability":0.0,"rank":4},{"product_id":"target_ecue_fin","product_name":"e-Account","probability":0.0,"rank":5},{"product_id":"target_nomina","product_name":"Payroll","probability":0.0,"rank":6},{"product_id":"target_nom_pens","product_name":"Pension Payroll","probability":0.0,"rank":7}],"latency_ms":29.04534339904785,"model_version":"20260116"}


Test Api

python -m src.api.test_api --sample --limit 100 --sleep 0.1

## Prometheus Metrics

The API exposes the following custom metrics at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `recommender_requests_total` | Counter | Total requests by endpoint and status |
| `recommender_prediction_latency_seconds` | Histogram | Prediction latency distribution |
| `recommender_recommendations_count` | Histogram | Number of recommendations returned |
| `recommender_top_probability` | Histogram | Top recommendation probability |
| `recommender_inprogress_requests` | Gauge | Currently processing requests |
| `recommender_validation_errors_total` | Counter | Validation errors by field |
| `recommender_batch_size` | Histogram | Batch prediction sizes |

## Grafana Dashboard

A pre-configured dashboard is available in Grafana showing:

- Request rate (success/sec)
- P95 latency
- In-progress requests
- Success rate
- Request rate by status over time
- Latency percentiles (p50, p90, p95, p99)
- Validation errors by field
- Top recommendation probability distribution

## File Structure

```
src/api/
├── main_api.py              # FastAPI application
├── get_prediction.py        # Model handler
├── validate_query.py        # Request validation
├── schemas.py               # Pydantic schemas
├── test_api.py              # API testing
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service orchestration
├── prometheus.yml           # Prometheus configuration
├── .env.example             # Environment template
├── README_API.md            # This file
└── grafana/
    ├── provisioning/
    │   ├── datasources/
    │   │   └── prometheus.yml
    │   └── dashboards/
    │       └── dashboard.yml
    └── dashboards/
        └── recommender.json
```

## Required Features

The API expects the following features in prediction requests:

### Core Features
- `fecha_dato`: Reference date (YYYY-MM-DD)
- `ncodpers`: Customer ID
- `age`: Customer age
- `customer_period`: Customer tenure

### Categorical Features
- `ind_empleado`, `pais_residencia`, `sexo`, `indrel`, `indrel_1mes`
- `tiprel_1mes`, `canal_entrada`, `cod_prov`, `segmento`

### Boolean Features
- `ind_nuevo`, `indresi`, `indfall`, `ind_actividad_cliente`
- `ind_*_fin_ult1`: Current product ownership (24 products)
- `ind_*_lag3`, `ind_*_lag6`: Lagged product ownership

### Numeric Features
- `renta`: Household income
- `n_products_lag3`, `n_products_lag6`: Number of products

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model is saved in `models/ovr_grouped_santander/`
2. **Port already in use**: Change ports in `.env` file
3. **Memory issues**: Adjust Docker memory limits in `docker-compose.yml`

### Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f recommender
docker-compose logs -f prometheus
docker-compose logs -f grafana
```

### Restart Services

```bash
# Restart a specific service
docker-compose restart recommender

# Rebuild and restart
docker-compose up -d --build recommender
```

## Development

### Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python -m src.api.main_api

# Or with uvicorn directly
uvicorn src.api.main_api:app --host 0.0.0.0 --port 8080 --reload
```


# Get access to the microservice front page
Open browser and navigate to http://127.0.0.1:8080

# Get acess to Prometheus
Open browser and navigate to http://localhost:9091/query

# Get acess to Grafana
Open browser and navigate to http://localhost:3001

# Get access to interactive API documentation
Open browser and navigate to http://127.0.0.1:8080/docs

# Get API metrics
Open browser and navigate to http://127.0.0.1:8080/metrics

# Stop and remove the containers
``` bash
docker stop sprint_3_stage_3_4_microservice_container
docker rm sprint_3_stage_3_4_microservice_container
docker stop prometheus                               
docker rm prometheus
docker stop grafana
docker rm grafana
```