from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

import sys
sys.path.append('../')  # 添加上一级目录到模块搜索路径
import detect_mask_image  # 导入上一级目录中的Python文件中的函数


app = Flask(__name__)

# 配置上传文件的保存目录
app.config['UPLOAD_FOLDER'] = 'uploads'

# 允许上传的文件类型
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
        # 在上传成功后调用检测口罩函数，并传递文件名参数
        generated_image_filename = detect_mask_image.prepare(filename)
        return render_template('result.html', image_filename=generated_image_filename)
        
    else:
        return 'Invalid file type'

if __name__ == '__main__':
    app.run(debug=True)
