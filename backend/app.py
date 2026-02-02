from flask import Flask, render_template, request, redirect, url_for,session
from tensorflow import keras 
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image  import img_to_array
import numpy as np
from PIL import Image
import os
import uuid
# ---------------------------
# App setup
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "../frontend/templates"),
    static_folder=os.path.join(BASE_DIR, "../frontend/static")
)

app.secret_key = "skinai_secret_key"

UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    

# ---------------------------
# Classes
# ---------------------------

CLASS_NAMES =  ['Cyst', 'blackheads', 'nodules', 'papules', 'pustules', 'whiteheads']

# ---------------------------
# LOAD MODEL
# ---------------------------

model = None
def load_acne_model():
    global model
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "model_with_inference_1.keras")
        model = tf.keras.models.load_model(MODEL_PATH)

        print("Model loaded succesfully..")
    except Exception as e:
        print(f"Model loading error---> {e}")
        model = None

load_acne_model()

# ---------------------------
# Prediction fucntion
# ---------------------------

def predict_acne(img_path):
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array)
    pre_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])*100
    return pre_class,confidence

# ---------------------------
# Routes
# ---------------------------

@app.route('/')
def home():
    return render_template("login.html")

@app.route("/login",methods = ['POST' , 'GET'])
def login():
    if request.method== "POST":

        username = request.form.get("username")
        role = request.form.get("role")

        if not role:
            return render_template("login.html" , error = "Please select the role")
    
        session['username'] = username
        session['user_type'] = role
        
        return redirect(url_for("upload_img"))
    
    return render_template("login.html")


@app.route("/upload",methods = ['POST' , 'GET'])
def upload_img():
    if 'user_type' not in session:
        return redirect(url_for("home"))
    
    return render_template("upload.html")


@app.route("/process",methods = ['POST' , 'GET'])
def process_image():
    if 'user_type' not in session:
        return redirect(url_for("home"))

    image = request.files.get("image")
    if not image:
        return render_template("upload.html", error="No image uploaded")

    filename = f"{uuid.uuid4().hex}.jpg"
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(img_path)

    acne_type, confidence = predict_acne(img_path)

    return render_template(
        "result.html",
        image_path=f"static/uploads/{filename}",
        acne_type=acne_type,
        confidence=confidence
    )

@app.route("/result",methods=['GET'])
def result():
    return render_template("result.html")
        

@app.route("/contact",methods=['GET'])
def contact():
    return render_template("contact.html")
        



if __name__ == "__main__":
    app.run(debug=True)
