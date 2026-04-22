#!/bin/bash
gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app --bind=0.0.0.0:8000