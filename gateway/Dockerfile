FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ../requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .
# Include visualization module
COPY ../visualization /app/../visualization

RUN mkdir -p /app/../outputs

EXPOSE 5000
CMD ["python", "app.py"]
