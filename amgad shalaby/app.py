from gettext import install
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.backend import argmax
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


model = load_model('BrainTumor_best_model.h5')


def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor"
	elif classNo==1:
		return "Yes Brain Tumor"


def getResult(img):
    imge = cv2.imread(img)
    img = Image.fromarray(imge)
    img = img.resize((64,64))
    img = np.array(img)
    #print(img)

    
    input_img = np.expand_dims(img,axis=0)
    predictions = model.predict(input_img)
    classes_x=argmax(predictions,axis=1)
    #print(classes_x)
    return classes_x


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('main_page.html')

@app.route('/admin_page', methods=['GET'])
def admin_page():
    return render_template('admin_page.html')

@app.route('/take_photo', methods=['GET'])
def take_photo():
    return render_template('take_photo.html')


@app.route('/predict', methods = ['POST'])
def predict():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    value=getResult(file_path)
    result=get_className(value) 
    return render_template('take_photo.html', prediction_text= result)


if __name__ == '__main__':
    app.run(debug=True)