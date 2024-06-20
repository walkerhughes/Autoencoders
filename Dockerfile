# Use the official FastAPI image from tiangolo
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11-slim AS base

# Set the working directory
WORKDIR /app

# Copy only the requirements file
COPY ./requirements_app.txt /app/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the rest of the application files
COPY ./app /app/app
COPY ./models/kde/kde_model_compressed.joblib /app/kde_model_compressed.joblib
COPY ./models/variational_autoencoder/variational_autoencoder_30_epochs.pth /app/variational_autoencoder_30_epochs.pth

# Expose the port FastAPI is running on
EXPOSE 80

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
