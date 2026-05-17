FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/results

ENV PYTHONPATH=/app

EXPOSE 8072

CMD ["python", "bee8_dashboard.py", "--host", "0.0.0.0", "--port", "8072"]
