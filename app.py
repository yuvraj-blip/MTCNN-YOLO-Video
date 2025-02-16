# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:21:15 2024

@author: KIIT
"""

from mtcnn import MTCNN
import os, time, cv2
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

cv2.setUseOptimized(True)

# Load YOLO model for object detection
yolo_model = YOLO('yolov8s.pt')


# Resizing the video frame to 1280x720 pixels
def resize_frame(frame, width=1280, height=720):
    return cv2.resize(frame, (width, height))


# Bilateral filtering for denoising the frames
def denoise_frame(frame):
    return cv2.bilateralFilter(frame, 15, 75, 75)


def detect_faces_mtcnn(frame, output_directory, frame_number, detector, face_metrics):
    results = detector.detect_faces(frame)
    face_metrics['total_faces'] += len(results)

    # Initialize variables to track the best face
    best_confidence = 0
    best_face = None
    best_face_box = None

    for i, result in enumerate(results):
        x1, y1, width, height = result['box']
        x2, y2 = x1 + width, y1 + height
        confidence = result['confidence']

        # Ensure coordinates are within frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if confidence > best_confidence:
            best_confidence = confidence
            best_face = frame[y1:y2, x1:x2]  # Crop the face
            best_face_box = (x1, y1, x2, y2)

    # Save the face with the highest confidence
    if best_face is not None and best_face.size > 0:  # Check if the cropped face is valid
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        face_filename = os.path.join(output_directory, f'best_face_frame_{frame_number}.jpg')
        cv2.imwrite(face_filename, best_face)
        print(f"Saved best face for frame {frame_number} at {face_filename}")

        # Draw a rectangle around the best face
        x1, y1, x2, y2 = best_face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box for best face

        # Update metrics
        face_metrics['confidences'].append(best_confidence)
    else:
        print(f"No valid face detected or best face crop failed for frame {frame_number}")

    return frame



# Detect objects using YOLO
def detect_objects(frame, object_metrics):
    results = yolo_model.predict(frame, stream=False)
    detected_objects = 0
    confidences = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls)
            conf = box.conf.item()
            confidences.append(conf)
            detected_objects += 1

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_name = yolo_model.names[class_id]
            cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    object_metrics['total_objects'] += detected_objects
    object_metrics['confidences'].extend(confidences)


def process_video(video_path, output_directory, skip_frames=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width, frame_height = 1280, 720

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file_name = f"processed_output_{timestamp}.mp4"
    output_path = os.path.join(output_directory, output_file_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    detector = MTCNN()

    frame_count = 0
    processed_count = 0
    total_time = 0
    face_metrics = {'total_faces': 0, 'confidences': []}
    object_metrics = {'total_objects': 0, 'confidences': []}

    with ThreadPoolExecutor(max_workers=4) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"End of video or cannot read the frame. Processed {processed_count} frames.")
                break

            # Skip frames for faster processing
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue

            start_time = time.time()
            frame_count += 1
            processed_count += 1
            print(f"Processing frame {frame_count}")

            # Resize and denoise the frame using a thread
            future = executor.submit(lambda f: denoise_frame(resize_frame(f, frame_width, frame_height)), frame)
            denoised_frame = future.result()

            # Detect faces and objects on the denoised frame
            detect_faces_mtcnn(denoised_frame, output_directory, frame_count, detector, face_metrics)
            detect_objects(denoised_frame, object_metrics)

            # Write the processed frame to output video
            out.write(denoised_frame)

            # Update processing time
            frame_processing_time = time.time() - start_time
            total_time += frame_processing_time
            print(f"Frame {frame_count} processing time: {frame_processing_time:.2f} seconds")

    cap.release()
    out.release()

    # Calculate average processing time per frame and average confidences
    avg_time_per_frame = total_time / processed_count if processed_count else 0
    avg_object_confidence = np.mean(object_metrics['confidences']) if object_metrics['confidences'] else 0
    avg_face_confidence = np.mean(face_metrics['confidences']) if face_metrics['confidences'] else 0

    print(f"\nProcessing complete. Output saved to {output_path}")
    print(f"Total frames processed: {processed_count}")
    print(f"Total frames skipped: {frame_count - processed_count}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average processing time per frame: {avg_time_per_frame:.2f} seconds")
    print(f"Total faces detected: {face_metrics['total_faces']}")
    print(f"Average face detection confidence: {avg_face_confidence:.2f}")
    print(f"Average object detection confidence: {avg_object_confidence:.2f}")



#Input and output path
video_path = r"C:/Users/KIIT/Desktop/College/IVA/IVA j component/iva_video.mp4"
output_directory = r"C:\Users\KIIT\Desktop\College\IVA\IVA j component\output" 


process_video(video_path, output_directory)
