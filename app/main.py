import base64
from PIL import Image
import io, json
import numpy as np
import cv2 as cv
from fastapi import FastAPI, WebSocket #, WebSocketDisconnect
from pydantic import BaseModel
import onnxruntime as rt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = FastAPI(title="Animal detection application",
    description="FastAPI app based on the ONNX model",
    version="0.2",)

# build the model from a config file and a checkpoint file
# This model loads just once when we start the API.
THRESHOLD = 0.3
CLASSES = ('Cat', 'Raccoon', 'Dog', 'Fox', 'Person', 'Mouse', 'Porcupine', 
               'Human_hand', 'Bird', 'Rabbit', 'Skunk', 'Squirrel', 'Deer', 'Snake')
model_path = './model/tiny_model.quant2.fix.onnx'
sess = rt.InferenceSession(model_path)

# define the Input class
class Input(BaseModel):
    base64str : str
    threshold : float = None

class InputTresh(BaseModel):
    threshold : float

def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img

def bytes_to_image(image_bytes):
    frame = cv.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    return frame

def inference_detector(session, img, thresh=THRESHOLD):
    if thresh is None: thresh = THRESHOLD 
    pred_classes, pred_boxes, pred_confidence = [], [], []
    h, w = img.shape[:2]
    # Image resize
    img = cv.resize(img, (416,416))
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    # Image normalization
    mean = np.float64(np.array([0,0,0]).reshape(1, -1))
    stdinv = 1 / np.float64(np.array([255.0,255.0,255.0]).reshape(1, -1))
    img = img.astype(np.float32)
    img = cv.subtract(img, mean, img)  
    img = cv.multiply(img, stdinv, img)  
    # Convert to [batch, c, h, w] shape
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    # Run model
    outputs = session.run(None, {'input': img})
    # Prepare results
    for box, cls in zip(outputs[0], outputs[1]):
        if box[-1] > thresh:
            pred_confidence.append(float(box[-1]))
            pred_classes.append(str(CLASSES[cls]))
            pred_boxes.append([int(box[0]*w/416), int(box[1]*h/416), int(box[2]*w/416), int(box[3]*h/416)])
    return {'boxes': pred_boxes, 
            'classes': pred_classes, 
            'confidence': pred_confidence}

@app.put("/set_threshold")
def set_threshold(d:InputTresh):
    '''
    Set the model's threshold
    '''
    global THRESHOLD
    THRESHOLD = d.threshold
    return {'status':f'threshold = {THRESHOLD}'}

@app.put("/predict")
async def get_predictionbase64(d:Input):
    '''
    FastAPI API will take a base 64 image as input and return a json object
    '''
    # Load the image
    img = base64str_to_PILImage(d.base64str)
    img = np.asarray(img)
    
    # get prediction on image
    result = inference_detector(sess, img.copy(), thresh=d.threshold)
    return result

@app.websocket("/predict")
async def websocket_endpoint(websocket: WebSocket):
    '''
    FastAPI API will take a bytes image as input and return a json object
    '''
    await websocket.accept()
    try:
        while True:
            bytes_data = await websocket.receive_bytes()
            img = bytes_to_image(bytes_data)
            result = inference_detector(sess, img.copy(), THRESHOLD)
            await websocket.send_json(result)
    except:# WebSocketDisconnect:
        # print('Disconnect')
        pass