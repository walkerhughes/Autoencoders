FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

COPY ./models/kde/kde_model_compressed.joblib /app/kde_model_compressed.joblib
COPY ./models/variational_autoencoder/variational_autoencoder.pth /app/variational_autoencoder.pth
COPY ./models/variational_autoencoder/variational_autoencoder_30_epochs.pth /app/variational_autoencoder_30_epochs.pth

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app