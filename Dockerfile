# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Render and other hosts set PORT at runtime
EXPOSE 8080

ENV PORT=8080

CMD gunicorn -w 1 -b 0.0.0.0:${PORT:-8080} main:app
