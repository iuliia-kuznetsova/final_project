# Bank Products Recommender API
Overview of a FastAPI-based recommendation service for bank products with Prometheus metrics and Grafana dashboards.

##  API Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Docker Compose Stack                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Recommender API Container                         │   │
│  │                              :8080                                       │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────────┐  │   │
│  │  │   FastAPI   │──▶│  Validator  │──▶│Model Handler│──▶│   OvR      │  │   │
│  │  │  (main_api) │   │(validate_   │   │(get_predict │   │  Models    │  │   │
│  │  │             │   │  query)     │   │   ion)      │   │(CatBoost)  │  │   │
│  │  └─────────────┘   └─────────────┘   └─────────────┘   └────────────┘  │   │
│  │         │                                                               │   │
│  │         │ /metrics                                                      │   │
│  │         ▼                                                               │   │
│  │  ┌─────────────────┐                                                    │   │
│  │  │ Prometheus      │                                                    │   │
│  │  │ Instrumentator  │                                                    │   │
│  │  └─────────────────┘                                                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                │                                                                │
│                │ scrape metrics                                                 │
│                ▼                                                                │
│  ┌─────────────────────┐            ┌─────────────────────┐                    │
│  │     Prometheus      │───────────▶│      Grafana        │                    │
│  │       :9090         │  query     │       :3000         │                    │
│  │  • Collect metrics  │            │  • Dashboards       │                    │
│  │  • Store time-series│            │  • Alerts           │                    │
│  │  • Retention: 15d   │            │  • Visualizations   │                    │
│  └─────────────────────┘            └─────────────────────┘                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Request Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           /predict Request Flow                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Client                                                                         │
│     │                                                                            │
│     │  POST /predict                                                             │
│     │  {customer_id, features, top_k}                                           │
│     ▼                                                                            │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │ Step 1: Request Reception (main_api.py)                                   │  │
│   │   • Parse JSON body                                                       │  │
│   │   • Validate against PredictionRequest schema                             │  │
│   │   • Start latency timer                                                   │  │
│   │   • Increment in-progress counter                                         │  │
│   └───────────────────────────────┬──────────────────────────────────────────┘  │
│                                   │                                              │
│                                   ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │ Step 2: Query Validation (validate_query.py)                              │  │
│   │   • Check customer_id format                                              │  │
│   │   • Verify all required features present                                  │  │
│   │   • Validate data types (int, float, bool, str)                          │  │
│   │   • Validate numeric ranges (age: 0-150, etc.)                           │  │
│   │   • If validation fails → 400 Bad Request                                │  │
│   └───────────────────────────────┬──────────────────────────────────────────┘  │
│                                   │                                              │
│                                   ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │ Step 3: Feature Preprocessing (get_prediction.py - ModelHandler)          │  │
│   │   • Convert request dict to DataFrame                                     │  │
│   │   • Add default values for missing optional features                     │  │
│   │   • Convert categorical columns to string                                │  │
│   │   • Convert boolean columns                                              │  │
│   │   • Align features to model's expected order                             │  │
│   └───────────────────────────────┬──────────────────────────────────────────┘  │
│                                   │                                              │
│                                   ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │ Step 4: Model Inference (get_prediction.py + modelling_ovr.py)            │  │
│   │                                                                           │  │
│   │   ┌─────────────────────────────────────────────────────────────────┐    │  │
│   │   │                    OvRGroupModel.predict_top_k()                 │    │  │
│   │   │                                                                  │    │  │
│   │   │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │    │  │
│   │   │   │ Frequent     │   │    Mid       │   │    Rare      │       │    │  │
│   │   │   │ Group Models │   │ Group Models │   │ Group Models │       │    │  │
│   │   │   │ (8 products) │   │ (7 products) │   │ (9 products) │       │    │  │
│   │   │   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘       │    │  │
│   │   │          │                  │                  │                │    │  │
│   │   │          └──────────────────┼──────────────────┘                │    │  │
│   │   │                             ▼                                   │    │  │
│   │   │                  ┌────────────────────┐                         │    │  │
│   │   │                  │ Collect 24 probs   │                         │    │  │
│   │   │                  │ Sort by probability│                         │    │  │
│   │   │                  │ Return top-K       │                         │    │  │
│   │   │                  └────────────────────┘                         │    │  │
│   │   └─────────────────────────────────────────────────────────────────┘    │  │
│   └───────────────────────────────┬──────────────────────────────────────────┘  │
│                                   │                                              │
│                                   ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │ Step 5: Response Building (main_api.py)                                   │  │
│   │   • Map product IDs to human-readable names                              │  │
│   │   • Calculate latency                                                    │  │
│   │   • Record Prometheus metrics                                            │  │
│   │   • Build PredictionResponse                                             │  │
│   │   • Decrement in-progress counter                                        │  │
│   └───────────────────────────────┬──────────────────────────────────────────┘  │
│                                   │                                              │
│                                   ▼                                              │
│   Client receives JSON response:                                                 │
│   {                                                                              │
│     "customer_id": "12345678",                                                  │
│     "recommendations": [                                                         │
│       {"product_id": "target_recibo", "product_name": "Direct Debit",           │
│        "probability": 0.85, "rank": 1},                                         │
│       ...                                                                        │
│     ],                                                                           │
│     "latency_ms": 45.2,                                                         │
│     "model_version": "20260116"                                                 │
│   }                                                                              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Model Loading Flow (Startup)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        API Startup / Model Loading                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Container Start                                                                │
│        │                                                                         │
│        ▼                                                                         │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │ 1. FastAPI Lifespan (main_api.py)                                         │  │
│   │    • Load environment variables                                           │  │
│   │    • Initialize logging                                                   │  │
│   └───────────────────────────────┬──────────────────────────────────────────┘  │
│                                   │                                              │
│                                   ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │ 2. Model Handler Initialization (get_prediction.py)                       │  │
│   │    • Create ModelHandler singleton                                        │  │
│   │    • Set model paths from env vars                                        │  │
│   └───────────────────────────────┬──────────────────────────────────────────┘  │
│                                   │                                              │
│                                   ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │ 3. Load OvR Model (modelling_ovr.py)                                      │  │
│   │    • Load metadata.json (features, thresholds, groups)                   │  │
│   │    • Load 24 CatBoost models (.cbm files)                                │  │
│   │    • Reconstruct OvR wrapper per group                                   │  │
│   │                                                                           │  │
│   │    models/ovr_grouped_catboost/                                          │  │
│   │    ├── metadata.json                                                      │  │
│   │    ├── frequent_metadata.json                                             │  │
│   │    ├── frequent_target_recibo.cbm                                         │  │
│   │    ├── frequent_target_cco_fin.cbm                                        │  │
│   │    ├── ...                                                                │  │
│   │    ├── mid_metadata.json                                                  │  │
│   │    ├── mid_target_*.cbm                                                   │  │
│   │    ├── rare_metadata.json                                                 │  │
│   │    └── rare_target_*.cbm                                                  │  │
│   └───────────────────────────────┬──────────────────────────────────────────┘  │
│                                   │                                              │
│                                   ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │ 4. Initialize Validator (validate_query.py)                               │  │
│   │    • Load feature schema                                                  │  │
│   │    • Set required features list                                           │  │
│   └───────────────────────────────┬──────────────────────────────────────────┘  │
│                                   │                                              │
│                                   ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │ 5. Update Prometheus Model Info                                           │  │
│   │    • Set model version tag                                                │  │
│   │    • Set n_features, n_products info                                      │  │
│   └───────────────────────────────┬──────────────────────────────────────────┘  │
│                                   │                                              │
│                                   ▼                                              │
│                         API Ready on :8080                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### `main_api.py` - FastAPI Application
Purpose: Main API application with endpoints and Prometheus instrumentation.
Endpoints:
- `/` - GET - API info
- `/health` - GET - Health check
- `/metrics` - GET - Prometheus metrics
- `/model/info` - GET - Model metadata
- `/predict` - POST - Single prediction
- `/docs` - GET - Swagger documentation
Prometheus Metrics Exposed:
- `recommender_requests_total` - Counter - Total requests by endpoint/status
- `recommender_prediction_latency_seconds` - Histogram - Prediction latency
- `recommender_recommendations_count` - Histogram - Recommendations per request
- `recommender_top_probability` - Histogram - Top recommendation confidence
- `recommender_inprogress_requests` - Gauge - Currently processing
- `recommender_validation_errors_total` - Counter - Validation failures by field

### `get_prediction.py` - Model Handler
Purpose: Load and manage the OvR model for inference.
Key Class: `ModelHandler`
Methods:
- `load_model()` - Load saved model from disk
- `predict(features, top_k)` - Generate recommendations
- `get_model_info()` - Return model metadata
- `health_check()` - Verify model is operational
Singleton Pattern: Model loaded once at startup, reused for all requests.

### `validate_query.py` - Request Validation
Purpose: Validate incoming prediction requests.
Key Class: `QueryValidator`
Validations Performed:
1. Customer ID format (string/int, max 20 chars)
2. Required features presence (75 features)
3. Data type validation (int, float, bool, str, date)
4. Numeric range validation (e.g., age: 0-150)

### `schemas.py` - Pydantic Models
Purpose: Define request/response schemas for API validation.
Key Models:
- `CustomerFeatures` - Feature input schema
- `PredictionRequest` - Full prediction request
- `PredictionResponse` - Prediction output
- `HealthResponse` - Health check response
- `ErrorResponse` - Error format


## Quick Start

### Build and Run with Docker Compose

```bash
# Build and start all services
docker compose -f src/api/docker-compose.yml up -d --build

# Check service status
docker compose -f src/api/docker-compose.yml ps

# View logs
docker compose -f src/api/docker-compose.yml logs -f recommender
```

### Rebuild After Code Changes

```bash
# Rebuild without cache
docker compose -f src/api/docker-compose.yml build --no-cache recommender
docker compose -f src/api/docker-compose.yml up -d

# Fix logs permissions if falls with error
sudo chown -R mle-user:mle-user ./logs/
```


## Service URLs
Recommender API - Main API: http://localhost:8080
API Documentation - Swagger UI: http://localhost:8080/docs
API Metrics - Prometheus metrics: http://localhost:8080/metrics
Prometheus - Metrics database: http://localhost:9090
Grafana - Dashboards (admin/admin): http://localhost:3000
MLflow - Experiment tracking: http://localhost:5000


## API Endpoints

### Health Check
```bash
curl -s http://localhost:8080/health | jq
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
```bash
curl -s http://localhost:8080/ | jq
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
```bash
curl -s http://localhost:8080/model/info | jq
```
Expected response:
```json
{
  "model_name": "ovr_grouped_catboost",
  "model_version": "20260116",
  "n_features": 75,
  "n_cat_features": 10,
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
```bash
curl -s -X POST http://localhost:8080/predict \
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
Expected response:
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
  "model_version": "20260116"
}
```

### Prometheus Metrics
```bash
curl -s http://localhost:8080/metrics | head -50
```


## Testing the API
### Run API Tests
```bash
# Generate sample data and run tests
python3 -m src.api.test_api --sample --limit 100 --sleep 0.1

# Test with custom configuration
python3 -m src.api.test_api \
  --host localhost \
  --port 8080 \
  --limit 50 \
  --batch-size 10 \
  --top-k 7
```


### Logs
```bash
# View all logs
docker compose -f src/api/docker-compose.yml logs -f

# View specific service logs
docker compose -f src/api/docker-compose.yml logs -f recommender
docker compose -f src/api/docker-compose.yml logs -f prometheus
docker compose -f src/api/docker-compose.yml logs -f grafana
```


### Restart Services
```bash
# Restart a specific service
docker compose -f src/api/docker-compose.yml restart recommender

# Rebuild and restart
docker compose -f src/api/docker-compose.yml up -d --build recommender
```


### Stop and Remove Containers
```bash
# Stop containers
docker compose -f src/api/docker-compose.yml down

# Stop and remove volumes
docker compose -f src/api/docker-compose.yml down -v
```