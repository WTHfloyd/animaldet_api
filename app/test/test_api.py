from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import base64
import requests,json
import sys

def PILImage_to_cv2(img):
    return np.asarray(img)

def drawboundingbox(img, boxes,pred_cls, rect_th=2, text_size=1, text_th=2):
    img = PILImage_to_cv2(img)
    class_color_dict = {}

    #initialize some random colors for each class for better looking bounding boxes
    for cat in pred_cls:
        class_color_dict[cat] = [np.random.randint(0, 255) for _ in range(3)]

    for i in range(len(boxes)):
        cv.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
                      (int(boxes[i][2]),int(boxes[i][3])),
                      color=class_color_dict[pred_cls[i]], thickness=rect_th)
        cv.putText(img,pred_cls[i], (int(boxes[i][0]), int(boxes[i][1])),  cv.FONT_HERSHEY_SIMPLEX, text_size, class_color_dict[pred_cls[i]],thickness=text_th) # Write the prediction class
    if img is None:
        sys.exit("Could not read the image.")
    cv.imshow("Display window", img[:,:,::-1])
    k = cv.waitKey(0)
    if k == ord("s"):
        cv.imwrite("result.png", img)


if __name__ == '__main__':
    image_path = 'app/test/dog.jpg'
    url = "http://127.0.0.1:8000/predict"
    # url = "http://localhost/predict"
    with open(image_path, "rb") as image_file:
        base64str = base64.b64encode(image_file.read()).decode("utf-8")
    payload = json.dumps({
        "base64str": base64str,
        "threshold": 0.4
        })
    response = requests.put(url,data = payload)
    data_dict = response.json()
    for index in range(len(data_dict['classes'])):
        print(f'{data_dict["classes"][index]}, conf={data_dict["confidence"][index]:.2f}, box={data_dict["boxes"][index]}')
    img = Image.open(image_path)
    drawboundingbox(img, data_dict['boxes'], data_dict['classes'])
