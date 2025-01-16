# UDKAST

import os
from pprint import pprint
from flask import Flask, request, flash, render_template
from werkzeug.utils import secure_filename

# Import the model
from models import RiceModel


UPLOAD_FOLDER =  os.path.join(os.path.dirname(__file__), "static")
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

NUM_CLASSES = 5  # Replace with the actual number of classes in your dataset
model = RiceModel(NUM_CLASSES)  # Initialize the model from models.py



@app.route("/", methods=['GET', 'POST'])
def upload(): 
    if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                return "No file"

            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return "No file selected"

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
            
            # Uploaded now infer
                label, precision = model.predict(path)
                return render_template('base.html', label=label, url=filename, precision="{:.2f}%".format(precision))

            return "File failed"
    else: 
        return render_template('base.html', label=None, url=None, precision=None)

@app.route("/json", methods=['POST'])
def json(): 
    # check if the post request has the file part
    if 'file' not in request.files:
        return "No file"

    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return "No file selected"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
    
        label, precision = model.predict(path)
        return {"label": label, "precision": precision}

    return "File failed"
    

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS