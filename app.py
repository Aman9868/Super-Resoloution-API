# Flask Application
from flask import Flask, request, render_template
import torch
import os
from PIL import Image
from diffusers import LDMSuperResolutionPipeline



app = Flask(__name__)
app.config['OUTPUT_FOLDER'] = 'static/output'
app.config['INPUT_FOLDER'] = 'static/input'

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/srgan_upscale", methods=["POST"])
def srgan_upscale_api():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"

    # load model and scheduler
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    input_image = request.files["image"]
    input_image.save(os.path.join(app.config['INPUT_FOLDER'], 'input_image.jpg'))
    image = Image.open(input_image)
    low_res_img = image.resize((128, 128))
    # run pipeline in inference (sample random noise and denoise)
    upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
    # save image
    upscaled_image.save(os.path.join(app.config['OUTPUT_FOLDER'], 'output_image.jpg'))
    return render_template('predict.html')


app.run(debug=True)





