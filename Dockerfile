# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set environment variables to avoid interaction during installation
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.10 python3.10-distutils python3-pip python3.10-venv libgl1-mesa-glx v4l-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Grant access to the video devices
RUN usermod -aG video www-data

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker's caching mechanism
COPY requirements.txt /app/

# Create a virtual environment and install dependencies
RUN python3.10 -m venv venv && \
    ./venv/bin/pip install --upgrade pip && \
    ./venv/bin/pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose the port your Flask app runs on
EXPOSE 5000

# Specify the command to run your application
CMD ["./venv/bin/python", "app.py"]
