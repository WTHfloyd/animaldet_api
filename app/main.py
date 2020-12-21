import base64
from PIL import Image
import io, json
import numpy as np
import cv2 as cv
from fastapi import FastAPI, WebSocket #, WebSocketDisconnect
from pydantic import BaseModel
import onnxruntime as rt
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = FastAPI(title="Animal detection application",
    description="FastAPI app based on the ONNX model",
    version="0.2",)

# build the model from a config file and a checkpoint file
# This model loads just once when we start the API.
THRESHOLD = 0.3
NMS_THRESHOLD = 0.5
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

def NMS(boxes, overlapThresh):
    """The fast implimentation Non Max Suppression algo (NMS) 
    with sorting by height
    
    Args:
        boxes <numpy.ndarray>:  boxes for algo, shape [N, 5]
                 a box format:  [x0,y0,w,h, confidence]
        overlapThresh <float>:   a treshold of overlapping

    Returns:
        boxes <np.ndarray>:     the boxes after the NMS, shape [N, 5]
        indexes <list[int]>:    indexes of choosing boxes
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    w = boxes[:,2]
    h = boxes[:,3]
    x2 = x1 + w
    y2 = y1 + h
    # compute the area of the bounding boxes 
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # grab the probability of the bounding boxes 
    # and sort the bounding boxes by the probabilities
    prob = boxes[:,-1]
    idxs = np.argsort(prob)
    # idxs = [i for i in range(len(boxes)-1,-1,-1)]
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int"), pick

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
    # calculate NMS algorithm
    _, indexes = NMS(outputs[0], NMS_THRESHOLD)
    # Prepare results
    for i, (box, cls) in enumerate(zip(outputs[0], outputs[1])):
        if i in indexes:
            if box[-1] > thresh:
                pred_confidence.append(float(box[-1]))
                pred_classes.append(str(CLASSES[cls]))
                pred_boxes.append([int(box[0]*w/416), int(box[1]*h/416),\
                                   int(box[2]*w/416), int(box[3]*h/416)])
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
    t0 = time.time()

    # Load the image
    img = base64str_to_PILImage(d.base64str)
    img = np.asarray(img)
    
    # get prediction on image
    result = inference_detector(sess, img.copy(), thresh=d.threshold)

    # get hte lead time
    lead_time = round(time.time() - t0, 3)
    result.update({'time': lead_time})
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
            t0 = time.time()
            img = bytes_to_image(bytes_data)
            result = inference_detector(sess, img.copy(), THRESHOLD)
            lead_time = round(time.time() - t0, 3)
            result.update({'time': lead_time})
            await websocket.send_json(result)
    except:# WebSocketDisconnect:
        # print('Disconnect')
        pass