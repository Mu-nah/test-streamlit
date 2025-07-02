# Use a slim Python base
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Streamlit entry point
CMD ["streamlit", "run", "customer_asset.py", "--server.port=8501", "--server.enableCORS=false"]
