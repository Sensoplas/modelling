FROM python:3.8
run pip install --no-cache-dir matplotlib joblib sklearn numpy
COPY ./UVIndexModel.joblib /app/
COPY ./scripts/getUVIndex.py /app/
ARG buildTime_MODELSIZE='default'
ENV MODELSIZE=$buildTime_MODELSIZE
ENTRYPOINT ["python", "/app/getUVIndex.py"]