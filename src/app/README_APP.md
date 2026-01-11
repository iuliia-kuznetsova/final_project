'''
  Building and running the FastAPI microservice, Prometeus monitoring and Grafana vizualization inside a Docker container using Docker Compose.
'''

# Prerequisites
- docker installed and running  
- the root directory of the project: `/home/mle-user/mle_projects/mle-project-sprint3-v001` 
- `.env` file located in `services/.env`
- `requirements.txt` file located in the project root
- trained model file is at `services/models/best_model_hyperparameter_optimization.pkl`
- dockercompose `docker-compose.yaml` in `services/`
- dockerfile `Dockerfile_stage_3_4` in `services/`
- api script `main_stage_3_4.py` in `services/ml_service`

# Run all following commands from the root directory 
# Change working directory to the root, if needed
```bash
cd mle_projects/mle-project-sprint3-v001
```

# Build and start all services
``` bash
docker compose -f services/docker-compose.yaml up --build -d
```
# Forward these ports manually
8080, 9091, 3001

# Check containers running
``` bash
  docker ps
```

# Check the container logs
``` bash
  docker logs sprint_3_stage_3_4_microservice_container
  docker logs prometheus
  docker logs grafana
```

# Example of a Curl request to microservice
```bash
  curl -X POST "http://127.0.0.1:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "apartment_id": "apartment_test",
    "model_params": {
      "f1": 3.0,
      "f2": 0.0,
      "f5": 2.0,
      "f6": 3.0,
      "f7": 3.0,
      "f8": 3.0,
      "f9": 3.0,
      "f15": 4.0,
      "f17": 4.0,
      "f19": 4.0,
      "f20": 3.0,
      "f21": 1.0,
      "f23": 2.0,
      "f24": 4.0,
      "f25": 4.0,
      "f26": 3.0,
      "f27": 3.0,
      "f28": 3.0,
      "f29": 0.0,
      "f31": 2.0,
      "f32": 3.0,
      "f33": 4.0,
      "f34": 3.0,
      "f35": 3.0,
      "f36": 3.0,
      "f39": 4.0,
      "f41": 4.0,
      "f43": 4.0,
      "f44": 2.0,
      "f45": 4.0,
      "f47": 3.0,
      "f48": 2.0,
      "f49": 3.0,
      "f50": 2.0,
      "f51": 4.0,
      "f52": 3.0,
      "f53": 4.0,
      "f54": 3.0,
      "f55": 3.0,
      "f56": 3.0,
      "f57": 4.0,
      "f58": 3.0,
      "f59": 3.0,
      "f60": 3.0,
      "f61": 3.0,
      "f62": 3.0,
      "f63": 3.0,
      "f64": 3.0,
      "building_type": 2,
      "build_decade": 2010
    }
  }'
```

# Expected response format
{
  'apartment_id': str,
  'price': float
}

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