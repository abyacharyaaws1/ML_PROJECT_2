# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set environment variables to avoid buffering of output
ENV PYTHONUNBUFFERED=1

# Create and set the working directory
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . /app/

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Expose port 5000 for the Flask application
EXPOSE 5000

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
