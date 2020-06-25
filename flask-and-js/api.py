from flask import Flask, render_template,redirect,request,Blueprint,jsonify
from flask_cors import CORS
import numpy as np
import base64
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from io import BytesIO
import os

# エラー発生のため、GPU使用無し
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# API実装
app = Flask(__name__)
CORS(app)
# モデルデータの読み込み、globalに作成
global model
model = keras.models.load_model('mnist_model.h5')
model._make_predict_function()
graph = tf.Graph()

# 接続時、画面表示
@app.route('/')
def init():
    return render_template('AIdrawing.html')

# POST　画像データをもとに数字認識の結果を返す
@app.route('/postImage', methods=['POST'])
def postImage():
    # POSTより、B64形式で取得
    image_str = request.form['image']
    # バイナリコードをPNGに変換、グレースケール化を行う
    image = Image.open(BytesIO(base64.b64decode(image_str))).convert('L')
    img_gray = np.array(image).reshape(1,28,28)
    # 白黒反転していたため、対応
    img_gray = (255 - img_gray) / 255.0

    # 学習済みモデルでの予想
    pred = model.predict(img_gray)
    ans = np.argmax(pred, axis=1)
    print(ans)
    # 結果をJson形式にて返す
    return jsonify({'output' : '答えは、{} です。'.format(ans)})

# magic
if __name__ == '__main__':
    app.run(debug=True)