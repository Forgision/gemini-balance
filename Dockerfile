FROM python:3.10-slim

WORKDIR /app

# Copy the entire application
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV API_KEYS='[]'
ENV ALLOWED_TOKENS='[]'
ENV TZ='Asia/Kolkata'

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
