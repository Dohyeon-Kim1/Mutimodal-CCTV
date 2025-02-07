import base64
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from flask import Flask, request, jsonify
from models import build_vast

app = Flask(__name__)

vast = build_vast()
vast.to("cuda", dtype=torch.float16)
vast.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711))
    ]
)

@app.route('/receive', methods=['POST'])
def receive_data():
    data = request.json
    arrs = np.frombuffer(base64.b64decode(data["data"]), dtype=data["dtype"]).reshape(data["shape"])[:,:,:,::-1]
    imgs = [Image.fromarray(arr) for arr in arrs]
    vision_pixels = torch.stack([transform(img) for img in imgs], dim=0).unsqueeze(0).to("cuda", dtype=torch.float16)

    with torch.inference_mode():
        caption = vast({"vision_pixels": vision_pixels}, task="cap%tv", compute_loss=False)["generated_captions_tv"][0]
    return jsonify({"caption": caption})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)