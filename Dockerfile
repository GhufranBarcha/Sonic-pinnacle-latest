# Use an official Python 3.12 image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the local files to the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && pip install --default-timeout=100 -r requirements.txt

# Expose the port that Gradio runs on
EXPOSE 7860

# Command to run the application
CMD ["python3", "app.py"]
