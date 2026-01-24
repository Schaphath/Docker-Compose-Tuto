ARG PYTHON_VERSION="3.12"

FROM python:${PYTHON_VERSION}-slim

LABEL maintainer="Matoki"
LABEL description="Application pour prédire le cancer de sein"


WORKDIR /app

COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt

# Copier le reste du code
COPY app.py .
COPY models ./models

# Exposition du port
EXPOSE 8000

# Lancement de l'application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]



