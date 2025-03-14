# Use the smallest possible base image
FROM python:3.8-alpine

# Install only essential system dependencies
RUN apk add --no-cache \
    cmake \
    libjpeg-turbo-dev \
    libpng-dev \
    openblas-dev \
    lapack-dev \
    g++ \
    && rm -rf /var/cache/apk/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install a lightweight prebuilt dlib version (no compilation)
RUN pip install --no-cache-dir dlib==19.24.2

# Remove dlib from requirements.txt to avoid reinstalling
RUN sed -i '/dlib/d' requirements.txt

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Start the Flask app with minimal resource usage
CMD ["gunicorn", "--workers=1", "--threads=2", "--timeout=0", "app:app"]
