# -------------------------
# BASE IMAGE (TensorFlow Compatible)
# -------------------------
FROM python:3.12-slim

# -------------------------
# ENVIRONMENT CONFIG
# -------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------
# SYSTEM DEPENDENCIES
# -------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libx11-6 \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# WORKDIR
# -------------------------
WORKDIR /app

# -------------------------
# COPY PROJECT FILES
# -------------------------
COPY . /app

# -------------------------
# UPGRADE PIP & ENSURE WHEEL SUPPORT
# -------------------------
RUN pip install --upgrade pip setuptools wheel

# -------------------------
# INSTALL PYTHON DEPENDENCIES
# -------------------------
# Prefer binary wheels to avoid heavy source builds
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# -------------------------
# EXPOSE PORT
# -------------------------
EXPOSE 10000

# -------------------------
# START GUNICORN SERVER (Render)
# -------------------------
CMD ["gunicorn", "app_backup:app", "--bind", "0.0.0.0:10000", "--workers=2", "--threads=2", "--timeout=300"]
