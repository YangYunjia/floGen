from flask import Flask, render_template, request, jsonify, send_file, current_app
import os
import subprocess
import time
import json

app = Flask(__name__)

# establish the wing api instance at the beginning of the client
# later use it to predict wing results given input parameters
from flowvae.app.wing.wing_api import Wing_api
wing_api = Wing_api(device='cuda:0')

def process_test_input(inputs):
    '''
    predict wing results given the input

    paras:
    ===
    `inputs` (`list`) the input parameters (every element should be float)
        > ONLY containes the required values! (len = 8 + 21)

    returns:
    ===
    a `list` of CL, CD, CM
    '''
    wg = wing_api.predict(inputs)
    wg.plot_2d(['upper', 'full'], contour=9,  write_to_file=os.path.join(current_app.root_path, 'image', 'wg.png'))
    wg.lift_distribution()
    cl_array = wg.cl
    # print(f"Processing test input: {test_input}")
    # print(f"CL: {cl_array}")
    return list(cl_array)

@app.route('/')
def index():
    return render_template('index.html')

# handle AJAX requests and conduct inference
@app.route('/process_input', methods=['POST'])
def process_input():

    t1 = time.time()    # start time

    input_num = request.json['test_input']
    # fetch inputs
    inputs = [float(input_num[i]) for i in list(range(0, 8)) + list(range(9, 30))]

    # inference results
    cl_array = process_test_input(inputs)

    # 获取图片的修改时间
    image_path = os.path.join(current_app.root_path, 'image', 'wg.png')
    mod_time = os.path.getmtime(image_path)
    formatted_time = time.strftime('%Y%m%d%H%M%S', time.localtime(mod_time))  # 使用时间戳
    image_url = f'/static/image/wg.png?t={formatted_time}'  # 在URL后附加时间戳}

    t2 = time.time()

    return jsonify({
        'status': 'success', 
        'data': inputs,
        'cl_array': cl_array,
        'image_url': image_url,
        'inference_time': t2 - t1
    })
    
    
# 路由：提供图片
@app.route('/static/image/wg.png')
def get_image():
    return send_file('image/wg.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=False)
