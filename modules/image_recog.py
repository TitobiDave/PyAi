from craft_text_detector import Craft
import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from data.vectordb import vectordb

class image_recog:
    def __init__(self, model_path):
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)


    def read_image_text(self, image_path, df):
        image = Image.open(image_path).convert("RGB")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"running on {device}")
        self.model.to(device)  # Move model to GPU
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(device)
        generated_ids = self.model.generate(pixel_values, max_new_tokens=1000)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)
        vdb = vectordb()
        products = vdb.recommend_products(generated_text, top_k=5)
        return {"final": generated_text, "products":products}
