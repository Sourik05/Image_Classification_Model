#!/bin/bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.app:app
