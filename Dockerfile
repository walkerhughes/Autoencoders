# Use the official FastAPI image from tiangolo
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11-slim 

# Copy only the requirements file
COPY ./requirements_docker.txt /app/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the rest of the application files
COPY ./app /app/app
COPY ./models/variational_autoencoder/onnx_files/decoder_model_10e.onnx /app/decoder_model_10e.onnx
COPY ./models/variational_autoencoder/onnx_files/decoder_model_30e.onnx /app/decoder_model_30e.onnx
COPY ./models/variational_autoencoder/onnx_files/decoder_model_50e.onnx /app/decoder_model_50e.onnx

# Expose the port FastAPI is running on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
