FROM python:3.11-slim

WORKDIR /app

COPY dashboard/requirements.txt ./dashboard/requirements.txt
RUN pip install --no-cache-dir -r dashboard/requirements.txt

COPY dashboard/ ./dashboard/

ENV PYTHONUNBUFFERED=1

EXPOSE 10000

CMD ["sh", "-c", "streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port ${PORT:-10000}"]
