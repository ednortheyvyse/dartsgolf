# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt ./

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app will run on. Flask's default is 5000.
EXPOSE 5000

# Command to run the application using a production-grade WSGI server (Gunicorn)
# Gunicorn is already in your requirements.txt.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
