from flask import Flask, render_template, request, jsonify, send_file, current_app
import os
import subprocess
import time
import json

app = Flask(__name__)

# establish the wing api instance at the beginning of the client
# later use it to predict wing results given input parameters
from flowvae.app.wing.wing_api import Wing_api
wing_api = Wing_api(device='default')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@app.route('/display_sectional_airfoil', methods=['POST'])
def handle_display_sectional_airfoil():
    # 获取前端发送的JSON数据
    data = request.get_json()
    inputs = data['inputs']  # 获取输入的 inputs[9:] 部分

    # 调用 wing_api 的 display_sectional_airfoil 函数，生成图片
    image_file = 'static/image/display_sectional_airfoil.png'
    wing_api.display_sectional_airfoil(inputs, write_to_file=image_file)

    # 返回图片的URL，供前端使用
    return jsonify({"image_url": f"/static/image/display_sectional_airfoil.png"})

@app.route('/display_wing_frame', methods=['POST'])
def handle_display_wing_frame():
    # 获取前端发送的JSON数据
    data = request.get_json()
    inputs = data['inputs']  # 获取输入的 inputs[9:] 部分

    # 调用 wing_api 的 display_sectional_airfoil 函数，生成图片
    image_file = 'static/image/display_wing_frame.png'
    wing_api.display_wing_frame(inputs, write_to_file=image_file)

    # 返回图片的URL，供前端使用
    return jsonify({"image_url": f"/static/image/display_wing_frame.png"})

@app.route('/predict_wing_flowfield', methods=['POST'])
def handle_predict_wing_flowfield():
    # 获取前端发送的JSON数据
    data = request.get_json()
    inputs = data['inputs']  # 获取输入的 inputs[9:] 部分

    # 调用 wing_api 的 display_sectional_airfoil 函数，生成图片
    image_file = 'static/image/predict_wing_flowfield.png'
    wg = wing_api.predict(inputs)
    wg.lift_distribution()
    cl_array = wg.cl
    wg.plot_2d(['upper', 'full'], contour=9, write_to_file=image_file)

    # 返回图片的URL，供前端使用
    return jsonify({"image_url": f"/static/image/predict_wing_flowfield.png", "cl_array": cl_array.tolist()})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)