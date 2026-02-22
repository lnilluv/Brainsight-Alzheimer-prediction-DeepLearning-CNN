# BrainSight - Alzheimer Prediction Platform

BrainSight is an end-to-end medical imaging inference platform for Alzheimer stage classification from MRI scans.
It combines model serving, API exposure, reverse proxying, experiment tracking, object storage, and a user-facing upload interface in a single containerized deployment.
The project is designed to demonstrate production-oriented MLOps and backend engineering decisions, not only model training.

Primary site: https://www.brainsight.tech

## System Architecture

The platform is deployed as a multi-service Docker Compose stack:

- `traefik`: TLS termination and ingress routing.
- `nginx`: static website hosting.
- `fastapi`: inference API.
- `streamlit`: interactive upload UI.
- `mlflow`: model registry and tracking server.
- `minio`: S3-compatible artifact storage.
- `mysqldb`: MLflow metadata store.
- `postgresdb`: API database backend.

The FastAPI service follows a layered boundary split:

- Domain: prediction rules and output shape.
- Application: use cases and inference port contracts.
- Adapters: HTTP routes and MLflow/TensorFlow integration.
- Composition root: dependency wiring in app bootstrap.

See `docs/architecture.md` for a focused architecture map.

## Tech Stack By Layer

- Model and inference: TensorFlow, NumPy, Pillow.
- API and application layer: FastAPI, Uvicorn, Python.
- Experiment tracking: MLflow.
- Data and storage: PostgreSQL, MySQL, MinIO S3.
- Frontend/UI: Streamlit, Nginx.
- Networking and edge: Traefik, Docker Compose.

## Data and Inference Flow

1. A user uploads an MRI image from Streamlit.
2. Streamlit sends the file to FastAPI over HTTPS.
3. FastAPI preprocesses the image and executes model inference through the MLflow adapter.
4. The API maps prediction vectors to Alzheimer stage labels.
5. A structured JSON response is returned to the client UI.

## Deployment Topology

- Publicly exposed: Traefik (`80/443`) with host-based routing.
- Internal-only services: FastAPI, Streamlit, MLflow, MinIO, MySQL, PostgreSQL are exposed on the internal Docker network only.
- TLS: handled at the edge by Traefik.
- MLflow access: protected behind Traefik auth middleware.

## Security and Hardening

Recent hardening includes:

- Removal of tracked certificate state from git history and repository tip.
- Secret hygiene via ignore rules and environment templates (`containers/.env.example`).
- Conservative dependency pinning and vulnerability scanning.
- Pinned runtime images for deterministic builds.
- Streamlit security controls enabled (CORS and XSRF protection).
- Reduced service exposure to limit attack surface.

## Run Locally

From repository root:

```bash
cp containers/.env.example containers/.env
docker compose --env-file containers/.env -f containers/docker-compose.yml build
docker compose --env-file containers/.env -f containers/docker-compose.yml up -d
docker compose --env-file containers/.env -f containers/docker-compose.yml ps
```

Quick smoke checks (with host headers):

```bash
curl -k -H "Host: api.example.com" https://localhost/
curl -k -H "Host: streamlit.example.com" https://localhost/
curl -k -H "Host: mlflow.example.com" https://localhost/
```

Shutdown:

```bash
docker compose --env-file containers/.env -f containers/docker-compose.yml down
```

## Repository Structure

```text
containers/
  app/
    fastapi/
      app/
        domain/
        application/
        adapters/
    streamlit/
    mlflow/
    traefik/
    nginx/
docs/
  architecture.md
```

## Demonstration Material

- Project video: https://youtu.be/N35KYIUFiWk
- DemoDay presentation: https://youtu.be/cRNy1-rTXYg?t=2090
- Web app demo: https://www.youtube.com/watch?v=3anHg1pY6PQ

## Authors

- Laurent: [GitHub](https://github.com/lnilluv) | [LinkedIn](https://www.linkedin.com/in/laurent-vullin/)
- Feriel: [GitHub](https://github.com/feeMdj) | [LinkedIn](https://www.linkedin.com/in/ferielhamedi/)
- Alexon: [GitHub](https://github.com/Alexon1999) | [LinkedIn](https://www.linkedin.com/in/alexon-uthayakumar-9361221a2/)
- Yuliya: [GitHub](https://github.com/YuliyaSheichenka) | [LinkedIn](https://www.linkedin.com/in/yuliya-sheichenka-6568a653/)
- Haikel: [GitHub](https://github.com/haikel11) | [LinkedIn](https://www.linkedin.com/in/ha%C3%AFkel-bouzazza-140647256/)

## References

- Alzheimer's Dataset (Kaggle): https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images
- OASIS datasets: https://www.oasis-brains.org/
