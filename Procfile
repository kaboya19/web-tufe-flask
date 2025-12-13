web: gunicorn app:app --workers 4 --threads 2 --timeout 120 --worker-class sync --max-requests 1000 --max-requests-jitter 50 --bind 0.0.0.0:$PORT

