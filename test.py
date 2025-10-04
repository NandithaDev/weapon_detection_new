import os
from ultralytics import YOLO
import cv2

# Load your trained YOLOv8 model
model = YOLO("runs/detect/train3/weights/best.pt")

frames_folder = "bg"          # input frames
output_folder = "detected_frames" # folder to save frames with detections
os.makedirs(output_folder, exist_ok=True)



for img_file in os.listdir(frames_folder):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(frames_folder, img_file)
        
        # Run prediction
        results = model.predict(img_path, conf=0.25, imgsz=640)
        
        # Check if any objects were detected
        if len(results[0].boxes) > 0:
            # Annotate image with bounding boxes
            annotated_frame = results[0].plot()

            # Convert from RGB to BGR for OpenCV
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            # Save annotated frame
            save_path = os.path.join(output_folder, img_file)
            cv2.imwrite(save_path, annotated_frame)
            print(f"Saved detected frame: {img_file}")

print("✅ Detection complete. All frames with weapons saved in:", output_folder)
