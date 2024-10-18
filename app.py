import string
import secrets
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from utils import utils

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("html/index.html")

@app.route("/result", methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        image = request.files['image']
        model = request.form['model']
        conf = float(request.form['conf'])

        characters = string.ascii_letters
        file_name = ''.join(secrets.choice(characters) for _ in range(64))
        file_path = "static/images/upload/" + file_name + ".jpg"
        image.save(file_path)
        
        predicted_image = utils.predict_from_path(file_path, model, conf)

        result_path = file_path.replace("upload", "result")
        plt.imsave(result_path, predicted_image)
        
    return render_template("html/result.html", img_src=result_path)