# Emotion Detection Dashboard

🔗 **Live Demo**: [emotion.dhruvkpatel.dev](https://emotion.dhruvkpatel.dev)

An interactive Streamlit dashboard that detects facial emotions from images, using MediaPipe for face detection and a PyTorch model for emotion classification.

## Features
- Upload an image and get the predicted emotion for each detected face
- Face detection with MediaPipe
- Emotion classification with a PyTorch CNN

## Run locally with Docker
```bash
docker build -t emotion-dashboard .
docker run -p 8501:8501 emotion-dashboard
```
Then open http://localhost:8501.

## Deployment
Self-hosted on my own server (Oracle Cloud, Arm) in Docker, behind Caddy with automatic HTTPS.

## Author
Dhruv Patel · [dhruvkpatel.dev](https://dhruvkpatel.dev)
