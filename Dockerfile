# Use a lightweight Python 3.10 image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (to cache this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the FinBERT model during build so the app is instantly responsive later
RUN pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
# Copy your source code
COPY . .

# Create directories for data and models if they don't exist
RUN mkdir -p data models

EXPOSE 5000

# Start the Flask dashboard
CMD ["python", "app.py"]
