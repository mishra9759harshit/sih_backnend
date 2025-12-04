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
# UPGRADE PIP
# -------------------------
RUN pip install --upgrade pip

# -------------------------
# INSTALL PYTHON DEPENDENCIES
# -------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# EXPOSE PORT
# -------------------------
EXPOSE 10000

# -------------------------
# START GUNICORN SERVER (Render)
# -------------------------
CMD ["gunicorn", "app_backup:app", "--bind", "0.0.0.0:10000", "--workers=2", "--threads=2", "--timeout=300"]
