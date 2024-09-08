from flask import Flask, render_template, request, jsonify, send_file, current_app
import numpy as np
import os
import subprocess
import time
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# 处理AJAX请求并调用run.py中的函数
@app.route('/process_input', methods=['POST'])
def process_input():
    input_num = request.json['test_input']
    # 获取test_input
    test_input = np.take(input_num, list(range(0, 8)) + list(range(9, 30)))
    # change each value to float
    test_input = [float(i) for i in test_input]

    result = subprocess.run(
            ["python3", "run.py", json.dumps(test_input)],
            capture_output=True,
            text=True
        )

        # 获取子进程输出（cl_array）
    cl_array = result.stdout.split('\n')[-2]

    # 获取图片的修改时间
    image_path = os.path.join(current_app.root_path, 'image/wg.png')
    mod_time = os.path.getmtime(image_path)
    formatted_time = time.strftime('%Y%m%d%H%M%S', time.localtime(mod_time))  # 使用时间戳
    image_url = f'/static/image/wg.png?t={formatted_time}'  # 在URL后附加时间戳}

    return jsonify({
        'status': 'success',
        'data': test_input,
        'cl_array': cl_array,
        'image_url': image_url
    })
    
    
# 路由：提供图片
@app.route('/static/image/wg.png')
def get_image():
    return send_file('image/wg.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
