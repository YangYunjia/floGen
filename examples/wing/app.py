import base64
from io import BytesIO
from matplotlib.figure import Figure
from flask import Flask, render_template, request, jsonify

# establish the wing api instance at the beginning of the client
# later use it to predict wing results given input parameters
from flowvae.app.wing.wing_api import Wing_api
wing_api = Wing_api(saves_folder='saves', device='default')

app = Flask(__name__)

@app.route('/display_sectional_airfoil', methods=['POST'])
def handle_display_sectional_airfoil():
    # 获取前端发送的JSON数据
    data = request.get_json()
    inputs = data['inputs']  # 获取输入的 inputs[9:] 部分
    
    fig = Figure(figsize=(5, 2), dpi=60)
    ax = fig.subplots()
    ax = wing_api.display_sectional_airfoil(ax, inputs)
    
    buf = BytesIO()
    fig.savefig(buf, format="png")

    # 返回图片，供前端使用
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return jsonify({"image": f"data:image/png;base64,{data}"})

@app.route('/display_wing_frame', methods=['POST'])
def handle_display_wing_frame():
    # 获取前端发送的JSON数据
    data = request.get_json()
    inputs = data['inputs']  # 获取输入的 inputs[9:] 部分

    # 调用 wing_api 的 display_sectional_airfoil 函数，生成图片
    fig = Figure(figsize=(5, 5), dpi=60)
    ax = fig.add_subplot(projection='3d')
    ax = wing_api.display_wing_frame(ax, inputs)

    buf = BytesIO()
    fig.savefig(buf, format="png")

    # 返回图片，供前端使用
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return jsonify({"image": f"data:image/png;base64,{data}"})

@app.route('/predict_wing_flowfield', methods=['POST'])
def handle_predict_wing_flowfield():
    # 获取前端发送的JSON数据
    data = request.get_json()
    inputs = data['inputs']  # 获取输入的 inputs[9:] 部分

    # 调用 wing_api 的 display_sectional_airfoil 函数，生成图片
    wg = wing_api.predict(inputs)
    wg.lift_distribution()
    cl_array = wg.cl
    
    fig = Figure(figsize=(14, 10), dpi=100)
    wg._plot_2d(fig, ['upper', 'full'], contour=9, reverse_y=-1)
    
    buf = BytesIO()
    fig.savefig(buf, format="png")

    # 返回图片，供前端使用
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return jsonify({"image": f"data:image/png;base64,{data}", "cl_array": cl_array.tolist()})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)