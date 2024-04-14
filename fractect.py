import os
import shutil

import cv2
import dicom2jpg
import torch
# from flask import *
from flask import Flask, render_template, request,redirect
from PIL import Image
from torchvision.transforms import v2

from adjustimage import AdjustImage
from confidencelevels import get_conf_for_image
from fasterrcnn import load_faster_rcnn_model, run_one_image
from gradcam import run_gradcam
from traintest import load_resnext_model

app = Flask(__name__)
DETECTION_MODEL = None
CLASSIFCATION_MODEL = None
ALLOWED_EXTENSIONS = {'dcm', 'jpg', 'jpeg', 'png', 'dicom'}
TEMP_UPLOAD_FOLDER = "static/inputs"
CLASSIFICATION_TRANSFORMS = v2.Compose(
    [AdjustImage(), v2.Resize([256, 256]), v2.PILToTensor()])
DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
app.config['UPLOAD_FOLDER'] = TEMP_UPLOAD_FOLDER
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def setup():
    if os.path.exists("static/inputs/uploadedFileClassification.jpg"):
        os.remove("static/inputs/uploadedFileClassification.jpg")
    if os.path.exists("static/results/modelgradcamoutput.jpg"):
        os.remove("static/results/modelgradcamoutput.jpg")
    if os.path.exists("static/inputs/uploadedFileDetection.jpg"):
        os.remove("static/inputs/uploadedFileDetection.jpg")
    if os.path.exists("static/results/modeldetectionimage.jpg"):
        os.remove("static/results/modeldetectionimage.jpg")
    if (os.path.isfile("models/detection_model.pth")
            and os.path.isfile("models/classification_model.pth")):
        try:
            global DETECTION_MODEL
            DETECTION_MODEL = load_faster_rcnn_model(
                "models/detection_model.pth")
        except Exception as e:
            print("Error loading Detection Model.")
            print(e)
        try:
            global CLASSIFCATION_MODEL
            CLASSIFCATION_MODEL = load_resnext_model(
                DEVICE, "models/classification_model.pth")
        except Exception as e:
            print("Error loading Classification Model.")
            print(e)
    else:
        print("Model files missing")


@app.route("/fractect")
def main():
    if DETECTION_MODEL is not None and CLASSIFCATION_MODEL is not None:
        return render_template("fractect.html")
    else:
        return render_template("error.html")


@app.route("/")
def landing_page():
    return redirect("/home")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/documentation")
def documentation():
    return render_template("documentation.html")
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/classify", methods=['POST'])
def classify():
    if request.method == 'POST':
        print(request.files['classification-image'])
        if 'classification-image' not in request.files:
            # Better error catching later, but for now,
            # if the person doesn't upload something, it just reloads the page
            return render_template("fractect.html")
            # return redirect('/fractect', 400)
        file = request.files["classification-image"]
        if file.filename == '':
            return render_template("fractect.html")
            # return redirect('/fractect', 400)
        if file and allowed_file(file.filename):

            # have to save it locally to send it to the model
            if file.content_type == 'application/octet-stream':
                file.save(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileClassification.dcm"))
                np_array = dicom2jpg.dicom2img(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileClassification.dcm"))
                dicom_img = Image.fromarray(np_array)
                dicom_img.save(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileClassification.jpg"))
                os.remove(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileClassification.dcm"))
            elif file.content_type == 'image/png':
                # do png conversion
                file.save(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileClassification.png"))
                img_array = cv2.imread(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileClassification.png"))
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],
                                         "uploadedFileClassification.jpg"), img_array,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                os.remove(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileClassification.png"))
            elif file.content_type == 'image/jpeg' or file.content_type == 'image/jpg':
                file.save(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileClassification.jpg"))
            else:
                return render_template("fractect.html")
        image = Image.open(os.path.join(
            app.config['UPLOAD_FOLDER'], "uploadedFileClassification.jpg"))
        image = CLASSIFICATION_TRANSFORMS(image)
        image = image.type(torch.FloatTensor)
        confidence_results = list(get_conf_for_image(
            image.unsqueeze(0), CLASSIFCATION_MODEL, DEVICE))

        run_gradcam(CLASSIFCATION_MODEL, image)
        print(confidence_results)
        # result gets saved to the correct location
        return render_template("fractect.html", results=confidence_results)


@app.route("/detect", methods=['POST'])
def detect():
    if request.method == 'POST':
        print(request.form.get("use-prev"))
        if request.form.get("use-prev") is not None:
            print("use-prev")
            if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'],
                                           "uploadedFileClassification.jpg")):
                shutil.copyfile(os.path.join(app.config['UPLOAD_FOLDER'],
                                             "uploadedFileClassification.jpg"), os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileDetection.jpg"))

            else:
                return render_template("fractect.html")
        else:
            print("og image")

            if 'detection-image' not in request.files:
                # Better error catching later, but for now,
                # if the person doesn't upload something, it just reloads the page
                return render_template("fractect.html")
                # return redirect('/fractect', 400)
            file = request.files["detection-image"]
            if file.filename == '':
                return render_template("fractect.html")
                # return redirect('/fractect', 400)
            if file and allowed_file(file.filename):
                # file.save(os.path.join(app.config['UPLOAD_FOLDER'], "uploadedFileDetection"))
                print(file.content_type)
                # have to save it locally to send it to the model
            if file.content_type == 'application/octet-stream':
                # do dicom conversion
                file.save(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileDetection.dcm"))
                np_array = dicom2jpg.dicom2img(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileDetection.dcm"))
                dicom_img = Image.fromarray(np_array)
                dicom_img.save(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileDetection.jpg"))
                os.remove(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileDetection.dcm"))
            elif file.content_type == 'image/png':
                # do png conversion
                file.save(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileDetection.png"))
                img_array = cv2.imread(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileDetection.png"))
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],
                                         "uploadedFileDetection.jpg"), img_array,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                os.remove(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileDetection.png"))
            elif file.content_type == 'image/jpg' or file.content_type == 'image/jpeg':
                file.save(os.path.join(
                    app.config['UPLOAD_FOLDER'], "uploadedFileDetection.jpg"))
            print(":)")
        image_path = os.path.join(
            app.config['UPLOAD_FOLDER'], "uploadedFileDetection.jpg")
        threshold = float(request.form.get("detection-threshold"))
        categories = get_categories(request.form)
        detection_results_labels, detection_results_scores = run_one_image(
            DETECTION_MODEL, image_path, threshold, categories)
        return render_template("fractect.html", detection_results_labels=detection_results_labels,
                               detection_results_scores=detection_results_scores)

@app.route('/error')
def error():
    return render_template("error.html")
def get_categories(form):
    categories = []
    if form.get("boneanomaly") is not None:
        categories.append(1)
    if form.get("bonelesion") is not None:
        categories.append(2)
    if form.get("foreignbody") is not None:
        categories.append(3)
    if form.get("fracture") is not None:
        categories.append(4)
    if form.get("metal") is not None:
        categories.append(5)
    if form.get("periostealreaction") is not None:
        categories.append(6)
    if form.get("pronatorsign") is not None:
        categories.append(7)
    if form.get("axis") is not None:
        categories.append(8)
    if form.get("softtissue") is not None:
        categories.append(9)
    if form.get("text") is not None:
        categories.append(10)
    return categories


with app.app_context():
    setup()
