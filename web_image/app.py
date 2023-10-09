from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

import sys
sys.path.append('../')  # Add the parent directory to the module search path
import detect_mask_image  # Import functions from a Python file in the parent directory

app = Flask(__name__)

# Configure the upload file saving directory
app.config['UPLOAD_FOLDER'] = 'uploads'

# Allowable file types for upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Call the mask detection function after a successful upload and pass the filename as a parameter
        generated_image_filename = detect_mask_image.prepare(filename)
        return render_template('result.html', image_filename=generated_image_filename)

    else:
        return 'Invalid file type'

if __name__ == '__main__':
    app.run(debug=True)
