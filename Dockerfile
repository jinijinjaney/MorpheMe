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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install prebuilt dlib instead of compiling
RUN pip install dlib==19.24.2

# Copy project files
COPY . .

# Upgrade pip (optional)
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch manually from the correct index
RUN pip install --no-cache-dir torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Remove dlib from requirements.txt to prevent reinstallation
RUN sed -i '/dlib/d' requirements.txt

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Start Flask app
CMD ["gunicorn", "app:app"]
