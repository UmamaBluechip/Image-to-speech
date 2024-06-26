from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import os
from functions import generate_image_caption, convert_text_to_speech
import soundfile as sf

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
  uploaded_file = request.files['image']
  if uploaded_file.filename != '':
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(image_path)

    caption = generate_image_caption(image_path)

    audio_arr = convert_text_to_speech(caption)
    sf.write("static/audio.wav", audio_arr, model_tts.config.sampling_rate)

    img = Image.open(image_path)
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_encoded = img_io.getvalue().encode('base64')

    return render_template('result.html', caption=caption, audio_path="audio.wav", image=img_encoded)
  else:
    return render_template('index.html', message="No image uploaded!")

if __name__ == '__main__':
  app.run(debug=True)
