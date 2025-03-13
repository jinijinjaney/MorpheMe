# Use Python 3.8 base image
FROM python:3.8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libpng-dev \
    libjpeg-dev \
    build-essential \
    libboost-python-dev \
    libboost-thread-dev \
    python3-dev \
    git \
    libdlib-dev \
    python3-dlib \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch manually from the correct index
RUN pip install --no-cache-dir torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Instead of pip install dlib, use the system dlib with a symlink approach
# This creates a symlink from the system dlib to the Python site-packages
RUN ln -s /usr/lib/python3/dist-packages/dlib* /usr/local/lib/python3.8/site-packages/

# Remove dlib from requirements.txt to prevent reinstallation
RUN sed -i '/dlib/d' requirements.txt

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Start Flask app
CMD ["gunicorn", "app:app"]