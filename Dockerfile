ARG PYTHON_VERSION="3.12"
FROM python:${PYTHON_VERSION}-slim

LABEL maintainer="Matoki"
LABEL description="Application pour prédire le cancer de sein"

WORKDIR /app
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY app.py .
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
