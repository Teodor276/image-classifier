# main.py

# uvicorn main:app --host 0.0.0.0 --port $PORT


from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import util
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageData(BaseModel):
    image_base64: str


@app.post("/classify_image/")
async def classify_image(data: ImageData):
    result = util.classify_image(image_base64_data=data.image_base64)
    return JSONResponse(content=result)
