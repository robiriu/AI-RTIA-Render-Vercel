import os
import io
import base64
from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageDraw
from io import BytesIO
from inference_sdk import InferenceHTTPClient
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI()

# Roboflow API Client
api_key = "TEHGbI0CKnrNwwzAMJPl"  # your API key
model_id = "palm-tree-detection-yr8yg/4"     # your Roboflow model

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=api_key
)

# Enable CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resize image before uploading
def resize_image(image: Image.Image, max_size=(1024, 1024)):
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

# Convert to base64 for Roboflow
def image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/detect/")
async def detect_palm_trees(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = resize_image(image)

        # Send image to Roboflow
        image_b64 = image_to_base64(image)
        result = await CLIENT.infer_async(image_b64, model_id=model_id)

        # Draw bounding boxes
        draw = ImageDraw.Draw(image)
        for i, pred in enumerate(result["predictions"]):
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            draw.text((x1 + 4, y1 + 4), f"Palm #{i+1}", fill="yellow")

        # Convert back to base64 for frontend
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        final_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "count": len(result["predictions"]),
            "image": final_image_b64
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
