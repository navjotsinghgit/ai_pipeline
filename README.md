# ai_pipeline


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
