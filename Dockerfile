# Use the official FastAPI image from tiangolo
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11-slim 

# Copy only the requirements file
COPY ./requirements_app.txt /app/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the rest of the application files
COPY ./app /app/app
COPY ./models/variational_autoencoder/onnx_files/decoder_model_30e.onnx /app/decoder_model_30e.onnx

# Expose the port FastAPI is running on
EXPOSE 80

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
