# Smart Personal Safety Device: Real-Time Video Analytics with Multi-Threaded Face & Object Detection

This software implementation is a part of a broader project which implements a multifunctional personal safety device. The device automatically transmits GPS coordinates and video footage to a registered phone number—either when a button is pressed or when specific events (such as sudden shocks, elevated stress, or abnormal heart rates) are detected. It processes video streams to identify objects and faces, compiling a detailed report with timestamps and location data for each detected event. This report, along with a video link, is then sent to the designated phone number.

- The project is aimed to increase the personal safety of people especially women who are more vulnerable to safety issues.

  <h3>Basic architecture of complete project</h3>

![image](https://github.com/user-attachments/assets/226abd7f-c841-4aae-988f-e91afd8c9828)

The softwareimplementation  integrates:

1.  MTCNN for robust face detection (selecting the best face per frame)
2.  YOLOv8 for accurate object detection,
3.  OpenCV for frame processing (resizing, denoising, annotation), and
4.  ThreadPoolExecutor to optimize processing speed.
   - The system processes video frames, draws bounding boxes around detected objects and faces, and outputs a processed video along with detailed metrics (e.g., average processing time, detection confidences).
