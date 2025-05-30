from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from PIL import Image
import cv2
import io

app = FastAPI()

# Monta archivos estáticos y templates
templates = Jinja2Templates(directory="templates")

# Carga el modelo una sola vez
model = joblib.load("modelo_knn_mnist.pkl")

def preprocess_image(file):
    img = Image.open(file).convert('L')
    img_np = np.array(img)

    if img_np.mean() > 127:
        img_np = 255 - img_np

    _, img_thresh = cv2.threshold(img_np, 30, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(img_thresh)
    x, y, w, h = cv2.boundingRect(coords)
    digit_cropped = img_thresh[y:y+h, x:x+w]
    digit_resized = cv2.resize(digit_cropped, (20, 20), interpolation=cv2.INTER_AREA)

    digit_final = np.zeros((28, 28), dtype=np.uint8)
    start_x = (28 - 20) // 2
    start_y = (28 - 20) // 2
    digit_final[start_y:start_y+20, start_x:start_x+20] = digit_resized

    return digit_final.reshape(1, -1)

@app.get("/", response_class=HTMLResponse)
def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    file = io.BytesIO(contents)
    input_vector = preprocess_image(file)
    prediction = model.predict(input_vector)
    return {"prediction": int(prediction[0])}
