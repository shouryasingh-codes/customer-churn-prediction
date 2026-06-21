# Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port (Hugging Face uses 7860 by default)
EXPOSE 7860

# Start command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
