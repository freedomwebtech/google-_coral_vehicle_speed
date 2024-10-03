import cv2
from ultralytics import YOLO
from speed import SpeedEstimator

# Load YOLOv8 model
model = YOLO("best_full_integer_quant_edgetpu.tflite")
line_pts = [(0, 175), (1018, 175)]

# Initialize global variable to store cursor coordinates
with open("coco1.txt", "r") as f:
    class_names = f.read().splitlines()

speed_obj = SpeedEstimator(reg_pts=line_pts, names=class_names)

# Open the video file or webcam feed
cap = cv2.VideoCapture('vid.mp4')

# Set video writer with the same resolution as the frames
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1020, 500))

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video stream ended or cannot be read.")
        break

#    count += 1
#    if count % 3 != 0:  # Skip some frames for speed (optional)
#        continue

    frame = cv2.resize(frame, (1020, 500))

    # Perform object tracking
    tracks = model.track(frame, persist=True, imgsz=240)
    
    # Estimate speed and get the processed frame
    im0 = speed_obj.estimate_speed(frame, tracks)
    
    # Write the processed frame to the video file
    video_writer.write(im0)

    # Display the frame with YOLOv8 results
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()
