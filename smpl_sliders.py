from easymocap.mytools.reader import read_smpl
from easymocap.socket.base_client import BaseSocketClient
import os
import numpy as np
from flask import Flask, render_template, request
import socket
import time

def send_rand(client):
    import numpy as np
    N_person = 10
    datas = []
    for i in range(N_person):
        transl = (np.random.rand(1, 3) - 0.5) * 3
        kpts = np.random.rand(25, 4)
        kpts[:, :3] += transl
        data = {
            'id': i,
            'keypoints3d': kpts
        }
        datas.append(data)
    for _ in range(1):
        for i in range(N_person):
            move = (np.random.rand(1, 3) - 0.5) * 0.1
            datas[i]['keypoints3d'][:, :3] += move
        client.send(datas)
        time.sleep(0.005)
    client.close()

def send_dir(client, path, step):
    from os.path import join
    from glob import glob
    from tqdm import tqdm
    from easymocap.mytools.reader import read_keypoints3d
    results = sorted(glob(join(path, '*.json')))
    for result in tqdm(results[::step]):
        if args.smpl:
            data = read_smpl(result)
            client.send_smpl(data)
        else:
            data = read_keypoints3d(result)
            client.send(data)
        time.sleep(0.005)

"""
        "Rh": [
          [1.617, 0.186, 0.268]
        ],
        "Th": [
          [-0.401, 1.154, 1.081]
        ],
        "poses": [
          [0.000, 0.000, 0.000, -0.183, 0.163, 0.169, 0.011, -0.123, -0.172, 0.000, 0.000, 0.000, 0.169, 0.207, -0.045, -0.114, -0.104, 0.217, 0.000, 0.000, -0.000, -0.204, 0.213, 0.068, -0.254, -0.088, -0.075, 0.000, 0.000, 0.000, 0.000, -0.000, -0.000, 0.000, -0.000, -0.000, -0.055, 0.137, -0.052, 0.000, -0.000, 0.000, -0.000, 0.000, 0.000, 0.119, 0.138, -0.070, -0.068, 0.119, -1.355, -0.043, -0.129, 1.218, -0.044, -0.392, -0.066, -0.051, 0.358, 0.142, 0.000, -0.000, 0.000, 0.000, -0.000, -0.000, 0.000, -0.000, -0.000, 0.000, 0.000, -0.000]
        ],
        "shapes": [
          [-0.659, -0.218, 0.060, 0.180, 0.035, 0.024, 0.000, -0.014, -0.002, 0.000]
        ]
"""

poses = np.array([[ 0.   ,  0.   ,  0.   , -0.067,  0.093,  0.162, -0.147, -0.082,
        -0.096,  0.   ,  0.   , -0.   ,  0.022,  0.044, -0.073,  0.094,
        -0.049,  0.042, -0.   ,  0.   ,  0.   , -0.043,  0.136, -0.004,
        -0.004, -0.144,  0.024,  0.034,  0.   ,  0.   , -0.   ,  0.   ,
        -0.   ,  0.   ,  0.   , -0.   , -0.288,  0.39 ,  0.043,  0.   ,
         0.   ,  0.   , -0.   ,  0.   ,  0.001,  0.281,  0.353,  0.019,
        -0.03 , -0.1  , -0.368, -0.027,  0.071,  0.426, -0.018, -0.158,
         0.065, -0.01 ,  0.091,  0.013, -0.   ,  0.   ,  0.   ,  0.   ,
        -0.   , -0.   , -0.   , -0.   , -0.   , -0.   , -0.   ,  0.   ]], dtype=np.float32)

shapes = np.array([[ 0.188, -0.178,  0.054,  0.178,  0.027,  0.035, -0.021, -0.022, 0.018, -0.015]], dtype=np.float32)

Th = np.array([[0.42 , 0.285, 1.168]], dtype=np.float32)
Rh = np.array([[ 1.279, -1.263, -1.197]], dtype=np.float32)

smpl_data={'id': 0,
               'Rh': Rh,
               'Th': Th,
               'poses': poses,
               'shapes': shapes,
               'expression': np.zeros((0,0), np.float32)}
data = [smpl_data,]
client = []

app = Flask(__name__)

def_vals = np.hstack((data[0]['poses'],data[0]['shapes'],data[0]['Rh'],data[0]['Th']))
val_names = ["{}".format(i) for i in range(def_vals.size)]
vals = [float(def_vals[0,i]) for i,v in enumerate(val_names)]
print("dofs: {}".format(val_names))

html_code = """<html>
    <head></head>
    <body>
        <form method="POST" action="test">
"""
for i,n in enumerate(val_names):
    html_code += """
            <input type="range" min="-2.0" max="2.0" step="any" value={{{}}} name="{}" oninput="this.nextElementSibling.value = this.value; submit()"/>
            <output>{{{}:10.4f}}</output>
""".format(i,n,i)

html_code += """
            <input type="submit" value="submit" />
        </form>

    </body>
</html>"""

@app.route("/test", methods=["POST"])
def test():
    global vals
    for i,v in enumerate(val_names):
        vals[i] = float(request.form[v])
    global data

    poses_range = range(0,data[0]['poses'].size)
    shapes_range = range(poses_range.stop, poses_range.stop + data[0]['shapes'].size)
    Rh_range = range(shapes_range.stop, shapes_range.stop + data[0]['Rh'].size)
    Th_range = range(Rh_range.stop, Rh_range.stop + data[0]['Rh'].size)

    for i,v in enumerate(val_names):
        #def_vals = np.hstack((data[0]['poses'],data[0]['shapes'],data[0]['Rh'],data[0]['Th']))
        if i in poses_range:
            data[0]['poses'][0,i-poses_range.start] = vals[i]
        elif i in shapes_range:
            data[0]['shapes'][0,i-shapes_range.start] = vals[i]
        elif i in Rh_range:
            data[0]['Rh'][0,i-Rh_range.start] = vals[i]
        elif i in Th_range:
            data[0]['Th'][0,i-Th_range.start] = vals[i]
    client[0].send_smpl(data)
    return html_code.format(*vals)

@app.route('/')
def index():
    return html_code.format(*vals)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--smpl', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.host == 'auto':
        args.host = socket.gethostname()
    client.append(BaseSocketClient(args.host, args.port))

    app.run()
