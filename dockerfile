# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app files to the container
COPY app.py .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World
ENV PORT 8080
ENV SERVER_URL http://localhost:5006
ENV WEBAPP_ORIGIN searchengine1-aozashrb6q-uc.a.run.app


# Run the command to start gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
