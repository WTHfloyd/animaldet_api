import asyncio
import websockets
import cv2 as cv
import time

async def hello(bytes):
    # uri = 'ws://127.0.0.1:8000/predict' 
    uri = "ws://159.203.26.162/predict"
    async with websockets.connect(uri) as websocket:
        t0 = time.time()
        await websocket.send(bytes)
        data = await websocket.recv()
        print(data)
        print(time.time()-t0)


if __name__ == '__main__':
    image_path = 'app/test/dog.jpg'
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    s_success, im_buf_arr = cv.imencode(".jpg", img)
    byte_im = im_buf_arr.tobytes()
    asyncio.get_event_loop().run_until_complete(hello(byte_im))