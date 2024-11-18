from fastapi import FastAPI, Form, HTTPException
import subprocess
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from moviepy.editor import VideoFileClip
from fastapi import FastAPI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/save-image-name/")
async def save_image_name(image_name: str = Form(...)):
    file_path = "C:\\Users\\trong\\Desktop\\NoctisAI - Detection\\server\\image-name-file.txt"
    try:
        with open(file_path, "w") as file:
            file.write(image_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to file: {e}")

    script_path = "C:\\Users\\trong\\Desktop\\NoctisAI - Detection\\server\\potholes_detection_using_ssd.py"
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error executing script: {e}")

    return {"message": "Image name saved and script executed successfully"}

@app.get("/get-output-image/")
async def get_output_image():
    image_path = r"C:\Users\trong\Desktop\NoctisAI - Detection\server\Real-Time-Pothole-Detection\pothole_detection_single_image.png"
    return FileResponse(image_path, media_type="image/png")

@app.get("/get-output-video/")
async def get_output_video():
    video_path = r'C:\Users\trong\Desktop\NoctisAI - Detection\dataset\Potholes Detection Inference on Video\output_video_detection.webm'

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    try:
        return FileResponse(
            video_path,
            media_type="video/webm",
            headers={"Content-Disposition": "inline; filename=output_video.webm"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading video file: {e}")

@app.post("/execute-webcam-script/")
async def execute_webcam_script():
    script_path = r"C:\Users\trong\Desktop\NoctisAI - Detection\server\pothole-detection-camera\pothole-detection-yolov8.py"
    try:
        subprocess.run(["python", script_path], check=True)
        return {"message": "Webcam script executed successfully"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error executing webcam script: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")



