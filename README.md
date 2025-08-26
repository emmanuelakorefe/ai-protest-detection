# Campus Protest Detection (MVP)

Upload short videos, auto-detect crowd size with YOLOv8, blur faces for privacy, trigger alerts when
crowds exceed a threshold for a sustained duration, and deliver annotated video + CSV + chart
with signed links via AWS S3/SNS.

---

## Demo Screenshots

### Upload & Settings  
![Upload Screen](assets/src1.png)

### Processed Results  
![Processed Results](assets/src2.png)

### Crowd Size Trends  
![Crowd Trends](assets/src3.png)

---

## Features
- Crowd detection using YOLOv8
- Face blurring for privacy
- Configurable thresholds (people count + duration)
- Automatic alerts via AWS SNS (email, SMS, etc.)
- Secure file outputs to S3 (video, CSV, chart, JSON)
Open http://<server-ip>:5000

Security & Privacy

Faces blurred by default

No credentials in repo (env vars only)

S3 objects private; alerts send expiring presigned URLs
## Run (local/EC2)
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export AWS_REGION=eu-central-1
export SNS_TOPIC_ARN=arn:aws:sns:eu-central-1:123456789012:topic



export S3_BUCKET=your-bucket
python app.py
