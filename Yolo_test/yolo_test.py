from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference on an image
results = model("superbikes.jpeg")

# Show the first result (optional)
results[0].show()

# Save the prediction image
results[0].save(filename="output.jpg")

# Load and show the saved image using OpenCV
img = cv2.imread("output.jpg")
cv2.imshow("YOLOv8 Output", img)
cv2.waitKey(0)  # Wait until any key is pressed
cv2.destroyAllWindows()
