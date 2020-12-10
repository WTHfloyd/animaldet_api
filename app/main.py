import base64
from PIL import Image
import io, json
import numpy as np
import cv2 as cv
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as rt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = FastAPI()

# build the model from a config file and a checkpoint file
# This model loads just once when we start the API.
CLASSES = ('Cat', 'Raccoon', 'Dog', 'Fox', 'Person', 'Mouse', 'Porcupine', 
               'Human_hand', 'Bird', 'Rabbit', 'Skunk', 'Squirrel', 'Deer', 'Snake')
model_path = './model/yolov3.quant2.onnx'
sess = rt.InferenceSession(model_path)

# define the Input class
class Input(BaseModel):
    base64str : str
    threshold : float

def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img

def inference_detector(session, img):
    # Image resize
    img = cv.resize(img, (320,320))
    # Image normalization
    mean = np.float64(np.array([0,0,0]).reshape(1, -1))
    stdinv = 1 / np.float64(np.array([255.0,255.0,255.0]).reshape(1, -1))
    img = img.astype(np.float32)
    img = cv.subtract(img, mean, img)  
    img = cv.multiply(img, stdinv, img)  
    # Convert to [batch, c, h, w] shape
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    outputs = session.run(None, {'input': img})
    return outputs

@app.put("/predict")
def get_predictionbase64(d:Input):
    '''
    FastAPI API will take a base 64 image as input and return a json object
    '''
    # Load the image
    img = base64str_to_PILImage(d.base64str)
    img = np.asarray(img)
    h, w = img.shape[:2]
    
    # get prediction on image
    result = inference_detector(sess, img.copy())
    pred_classes, pred_boxes, pred_confidence = [], [], []
    for box, cls in zip(result[0], result[1]):
        if box[-1] > d.threshold:
            pred_confidence.append(float(box[-1]))
            pred_classes.append(str(CLASSES[cls]))
            pred_boxes.append([int(box[0]*w/320), int(box[1]*h/320), int(box[2]*w/320), int(box[3]*h/320)])

    return {'boxes': pred_boxes,
            'classes': pred_classes,
            'confidence': pred_confidence}

