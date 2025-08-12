# Gateway

FastAPI service exposing `/submit-dispute` and `/status/{id}` and emitting PoDP receipts.
Run with uvicorn:
```
uvicorn services.gateway.app:app --reload --host 0.0.0.0 --port 8000
```
