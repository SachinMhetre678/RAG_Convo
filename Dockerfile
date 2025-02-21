# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    swig \
    && rm -rf /var/lib/apt/lists/*  # Clean up package list to reduce image size

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt ./ 

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app's code into the container
COPY . . 

# Expose the port Flask runs on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
