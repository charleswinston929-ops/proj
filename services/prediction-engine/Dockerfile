FROM python:3.12-slim
WORKDIR /srv/services/prediction-engine
COPY services/prediction-engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/prediction-engine/ .
COPY data /srv/data
ENV ARTIFACT_PATH=/srv/artifacts/app_data.json
RUN useradd -m app && chown -R app /srv
USER app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
