from src import detect_faces
from PIL import Image
import sys
import logging
import os
import shutil
            

def detect_face(image_path, extend_factor=0.2):
    image = Image.open(image_path)
    bounding_boxes, landmarks = detect_faces(image, thresholds=[0.4, 0.5, 0.6])
    if bounding_boxes.shape[0] == 0:
        for degree in [90, -90, 180]:
            img = image.rotate(degree)
            bounding_boxes, landmarks = detect_faces(img, thresholds=[0.4, 0.5, 0.6])
            if bounding_boxes.shape[0] > 0:
                image = img
                break
        if bounding_boxes.shape[0] == 0:
            logging.warning(f"Failed to detect face on image {image_path}.")
            return None
    most_likely = bounding_boxes[:, 4].argmax()
    bounding_box = bounding_boxes[most_likely, :4]
    margin_w = (bounding_box[2] - bounding_box[0]) * extend_factor
    margin_h = (bounding_box[3] - bounding_box[1]) * extend_factor
    bounding_box[0] = max(0, bounding_box[0] - margin_w)
    bounding_box[1] = max(0, bounding_box[1] - margin_h * 1.6)
    w, h = image.size
    bounding_box[2] = min(w, bounding_box[2] + margin_w)
    bounding_box[3] = min(h, bounding_box[3] + margin_h)
    face = image.crop(bounding_box)
    return face

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    image_dir = os.path.abspath(os.path.expanduser(sys.argv[1]))
    output_dir = os.path.abspath(os.path.expanduser(sys.argv[2]))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    i = 0
    for sub_dir in os.listdir(image_dir):
        sub_path = os.path.join(image_dir, sub_dir)
        if os.path.isdir(sub_path):
            sub_output_path = os.path.join(output_dir, sub_dir)
            os.mkdir(sub_output_path)
            for name in os.listdir(sub_path):
                if name.endswith('.jpg'):
                    i += 1
                    path = os.path.join(sub_path, name)
                    output_path = os.path.join(sub_output_path, name)
                    face = detect_face(path)
                    if face is not None:
                        face.save(output_path)
                    if i % 100 == 0:
                        logging.info(f"Processed {i} images.")                  