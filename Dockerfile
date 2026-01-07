# Image Python 
FROM python:3.11-slim

# Empêche la création de fichiers .pyc et active l'affichage des logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Définit le dossier de travail 
WORKDIR /app

# Crée un utilisateur non-root AVANT d'installer les dépendances
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copie les dépendances en tant que root
COPY --chown=appuser:appuser requirements.txt .

# Passe à l'utilisateur non-root pour l'installation
USER appuser

# Met à jour pip et installe les dépendances
RUN pip install --user --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# Ajoute le répertoire des packages utilisateur au PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Copie tout le code de l'application
COPY --chown=appuser:appuser . .

# Expose le port 8000 (plus standard pour FastAPI)
EXPOSE 8000

# Lance le serveur FastAPI avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
