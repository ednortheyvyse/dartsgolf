# --- Build Stage ---
FROM python:3.11-slim as builder

WORKDIR /usr/src/app

# Install build dependencies
RUN pip install --upgrade pip

# Copy and install Python requirements
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /usr/src/app/wheels -r requirements.txt


# --- Final Stage ---
FROM python:3.11-slim

WORKDIR /usr/src/app

# Copy pre-built wheels and application code
COPY --from=builder /usr/src/app/wheels /wheels
COPY . .

# Install Python dependencies from wheels
RUN pip install --no-index --find-links=/wheels -r requirements.txt

# Expose the port Gunicorn will run on
EXPOSE 8000

# Run the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]