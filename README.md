# MTCNN

`pytorch` implementation of **inference stage** of face detection algorithm described in  
[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878).


## Some Improvements/Moifications in this fork

1. Adapt to Pytorch 1.0
2. Handle grayscale images.
3. Improve consistency. Return an empty numpy.ndarray when not finding faces. 
4. Add a script to align datasets with LFW-like structures.

## Align a dataset

```bash
python align_dataset.py <dataset dir> <output dir>
```
## Example
![example of a face detection](images/example.png)

## How to use it
Just download the repository and then do this
```python
from src import detect_faces
from PIL import Image

image = Image.open('image.jpg')
bounding_boxes, landmarks = detect_faces(image)
```
For examples see `test_on_images.ipynb`.

## Requirements
* pytorch 1.0
* Pillow, numpy

## Credit
This implementation is heavily inspired by:
* [pangyupo/mxnet_mtcnn_face_detection](https://github.com/pangyupo/mxnet_mtcnn_face_detection)  
