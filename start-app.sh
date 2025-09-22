#!/bin/bash

python scripts/run_dashboard.py &

uvicorn src.api.app:app --host 0.0.0.0 --port 9000