import io
import requests
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

HF_MODEL_PATH = 'line-corporation/clip-japanese-base-v2'
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
processor = AutoImageProcessor.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(HF_MODEL_PATH, trust_remote_code=True).to(device)

image = Image.open(io.BytesIO(requests.get('https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260').content))
image = processor(image, return_tensors="pt").to(device)
text = tokenizer(["犬", "猫", "象"]).to(device)

with torch.no_grad():
    image_features = model.get_image_features(**image)
    text_features = model.get_text_features(**text)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
# [[1., 0., 0.]]