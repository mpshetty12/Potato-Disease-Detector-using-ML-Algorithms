from fastapi import FastAPI, File, UploadFile
import uvicorn 
import tensorflow as tf


import numpy as np
from PIL import Image
from io import BytesIO

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8080"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("./models/1.h5")
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    return np.array(image)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    #bytes = await file.read()
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)
    predictions = MODEL.predict(img_batch)
    CLASS_NAMES[np.argmax(predictions[0])]
    CONFIDENSE = 100*np.max(predictions[0])
    return {
        "class": CLASS_NAMES[np.argmax(predictions[0])],
        "confidence": float(CONFIDENSE)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
