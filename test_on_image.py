from src import detect_faces
from PIL import Image
import sys

image_path = sys.argv[1]
image = Image.open(image_path)
bounding_boxes, landmarks = detect_faces(image, thresholds=[0.4, 0.5, 0.6])
print(bounding_boxes)
print(landmarks)
