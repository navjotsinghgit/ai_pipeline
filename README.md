# ai_pipeline: Retail Shelf Product Detection & Grouping


## Overview

Build a microservices-based AI pipeline that accepts retail shelf images, detects products using YOLOv8n, groups products by brand using color+texture clustering, and returns JSON with bounding boxes + brand group IDs — while also saving annotated visualization images.

### Project Structure

ai_pipeline/
│   ├── gateway/
│   │   ├── app.py               # Flask API Gateway
│   │   ├── templates/
│   │   │   └── index.html       # web UI
│   │   └── static/
│   │       └── style.css
│   ├── detection_service/
│   │   ├── app.py               # Detection microservice
│   │   └── detector.py          # YOLOv8n 
│   ├── grouping_service/
│   │   ├── app.py               # Grouping microservice
│   │   └── grouper.py           # Feature extraction + clustering
│   ├── visualization/
│   │   └── visualizer.py        # Annotated image generation
│   ├── outputs/                 # Saved visualization images
│   ├── requirements.txt
│   ├── docker-compose.yml
│   ├── README.md
│   └── run_all.sh               # Local start script



## Architecture

Browser (HTML UI)
     │  POST /analyze (multipart image)
     ▼
Flask API Gateway (port 5000)
     │  sends image bytes via ZeroMQ / HTTP
     ├──► Detection Microservice (port 5001)
     │        YOLOv8n (nano) — lightweight, fast
     │        Returns: [{bbox, confidence, class}]
     │
     └──► Grouping Microservice (port 5002)
              Takes detected crops → extracts features 
              Clusters with HDBSCAN or KMeans
              Returns: [{detection_id, brand_group_id}]
