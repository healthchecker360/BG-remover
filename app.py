from flask import Flask, render_template, request, send_file
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import gdown
import cv2

from model.u2net import U2NET  # from GitHub repo

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "u2net.pth"
MODEL_URL = "https://huggingface.co/xuebinqin/U-2-Net/resolve/main/u2net.pth"

# Download model weights if not present
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
net = U2NET(3, 1)
net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
net.eval()

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

def remove_background(image):
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        pred = net(img_tensor)[0][0]

    mask = pred.squeeze().cpu().numpy()
    mask = cv2.resize(mask, image.size)
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        image = Image.open(input_path).convert("RGB")
        mask = remove_background(image)

        image_np = np.array(image)
        result = cv2.bitwise_and(image_np, image_np, mask=mask)

        output_path = os.path.join(UPLOAD_FOLDER, "output.png")
        Image.fromarray(result).save(output_path)

        return send_file(output_path, as_attachment=True)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
