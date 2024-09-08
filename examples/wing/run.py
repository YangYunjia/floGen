from flowvae.app.wing.wing_api import Wing_api
import numpy as np
import os
import sys
import json
from matplotlib import pyplot as plt

def process_test_input(test_input):
    api = Wing_api(device='mps')
    wg = api.predict(np.array(test_input))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wg.plot_2d(['upper', 'full'], contour=9,  write_to_file = os.path.join(current_dir, 'image/wg.png'))
    wg.lift_distribution()
    cl_array = wg.cl
    # print(f"Processing test input: {test_input}")
    # print(f"CL: {cl_array}")
    return cl_array

if __name__ == "__main__":
    test_input = json.loads(sys.argv[1])  # 获取传入的test_input参数
    cl_array = process_test_input(test_input)
    print(json.dumps(cl_array.tolist()))


