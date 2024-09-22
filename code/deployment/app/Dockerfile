FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV API_HOST=api
ENV API_PORT=8000

CMD ["streamlit", "run", "app.py"]