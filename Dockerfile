# ==========================================
# STAGE 1: Builder (Installs heavy binaries)
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy only requirements first to efficiently leverage Docker layer caching
COPY requirements.txt .

# Install dependencies into a localized target directory
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==========================================
# STAGE 2: Runtime (Minimal execution footprint)
# ==========================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy only the compiled packages from the builder stage
COPY --from=builder /install /usr/local
COPY main.py .

# Run the python app immediately when the container spins up
CMD ["python", "main.py"]