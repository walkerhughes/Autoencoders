FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

COPY ./app/model/kde_model.joblib /app/kde_model.joblib

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app