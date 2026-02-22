# Architecture Overview

## System Map
- `nginx`: static portfolio website hosting.
- `fastapi`: prediction API for uploaded MRI images.
- `streamlit`: lightweight UI to upload images and call API.
- `mlflow`: model registry/tracking backend.
- `minio`, `mysqldb`, `postgresdb`: backing infrastructure.
- `traefik`: TLS termination and reverse proxy.

## FastAPI Service Layers

### Domain
- File: `containers/app/fastapi/app/domain/prediction.py`
- Contains label mapping and prediction result shaping.
- No framework, no database, no HTTP concerns.

### Application
- Files:
  - `containers/app/fastapi/app/application/ports.py`
  - `containers/app/fastapi/app/application/use_cases.py`
- Defines model inference port and use case orchestration.
- Depends only on domain contracts.

### Adapters
- Files:
  - `containers/app/fastapi/app/adapters/mlflow_model.py`
  - `containers/app/fastapi/app/adapters/http/routes.py`
- Implements external integration with MLflow/TensorFlow.
- Exposes HTTP routes via FastAPI.

### Composition Root
- File: `containers/app/fastapi/app/main.py`
- Wires model adapter to use case and registers routes.

## Security Posture
- TLS routed by Traefik.
- Streamlit CORS and XSRF protections enabled.
- Sensitive runtime certificate state is not tracked in git.
- Service images and Python dependencies are pinned.
- Internal services use `expose`; only reverse proxy publishes ports.
