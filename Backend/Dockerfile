# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy backend files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
