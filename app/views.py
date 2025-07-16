from flask import render_template, request
from flask import redirect, url_for
from PIL import Image
import os
from app.utils import pipeline_model

UPLOAD_FOLDER = 'static/uploads/'


def linkedin():
    return redirect('http://www.linkedin.com/in/piotr-karwacki-134579372')


def base():
    return render_template("base.html", title="Rozpoznawanie Twarzy - Strona Główna")


def index():
    return render_template("index.html", title="Rozpoznawanie Twarzy - Strona Główna")


def faceapp():
    return render_template("faceapp.html", title="Rozpoznawanie Twarzy - O Stronie")


def getwidth(path):
    img = Image.open(path)
    size = img.size  # width, height
    aspect = size[0] / size[1]  # width/height
    w = 300 * aspect
    return int(w)


def gender():

    if request.method == 'POST':
        f = request.files['image']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        # Processing
        w = getwidth(path)

        # Predictions (pass to pipeline model)
        pipeline_model(path, filename, color='bgr')

        return render_template("gender.html",
                               title="Rozpoznawanie Twarzy - Klasyfikacja Płci",
                               fileupload=True,
                               img_name=filename,
                               w=w,
                               text_1="Obraz oryginalny",
                               text_2="Obraz i predykcja")

    return render_template("gender.html",
                           title="Rozpoznawanie Twarzy - Klasyfikacja Płci",
                           fileupload=False,
                           img_name=None,
                           w=300,
                           text_1="",
                           text_2="")
