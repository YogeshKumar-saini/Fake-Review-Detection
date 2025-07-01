# Use a lightweight Python image
FROM python:3.11-slim

# Set environment variables to prevent stdin issues
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy dependency files first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Streamlit disables browser auto-opening inside Docker
ENV STREAMLIT_SERVER_HEADLESS=true

# Command to run your Streamlit app
# Adjust the path to your main streamlit app if needed
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
