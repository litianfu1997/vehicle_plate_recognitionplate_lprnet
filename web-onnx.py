import base64
import io
import onnxruntime as ort
import time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import plate_mnet_detect_test, plate_dict_test
from PIL import Image

app = Flask("plate-onnx")
CORS(app, supports_credentials=True)

# 初始化车牌识别模型
plate_dict_model_path = "lprnet.onnx"
plate_dict_ort_session = ort.InferenceSession(plate_dict_model_path)

# 初始化车牌检测模型
plate_detect_model_path = "mnet_plate.onnx"
plate_detect_ort_session = ort.InferenceSession(plate_detect_model_path)


@app.route('/api/test', methods=['GET'])
def test():
    print("hello world")
    return "hello world"


@app.route('/api/upload_img', methods=['POST'])
def upload_img():
    start = int(time.time() * 1000)
    imgs = request.files.getlist('file')
    obj_flag = request.form['obj_flag']
    base64_img = request.form['base64_img']
    plate_entity = []
    for f in imgs:
        # 读取图片
        filename = f.filename
        stream = f.stream.read()
        image = Image.open(io.BytesIO(stream))
        image = image.convert("RGB")  # 图片转为RGB格式
        # 图片转为BGR格式
        image = np.array(image)[:, :, ::-1]
        image = Image.fromarray(image)

        ximg = []
        # 判断是否开启车牌目标检测
        if obj_flag != 'false':
            bboxes = detector(image)
            array = np.array(image)
            if len(bboxes) > 0:
                for x1, y1, x2, y2 in bboxes:
                    ximg.append(Image.fromarray(array[y1:y2, x1:x2]))
        else:
            ximg.append(image)
        plates = lprnet_look(ximg)
        base64_data = "none"
        if base64_img == "true":
            base64_data = str(base64.b64encode(stream))[2:-1]
        plate = {"filename": filename, "plates": plates, "base64_data": base64_data}
        plate_entity.append(plate)
    end = int(time.time() * 1000)

    print("检出数：" + str(len(plate_entity)))
    print("耗时：", str(end - start) + " 毫秒")
    return jsonify({"code": 0, "msg": "succeed", "plate_entity": plate_entity})


# 车牌检测
def detector(img):
    if img is not None:
        bboxes = plate_mnet_detect_test.get_plate_pos(img, plate_detect_ort_session)
        return bboxes
    else:
        return []


# 车牌识别
def lprnet_look(imgs):
    plates = []
    for img in imgs:
        # 指定尺寸
        img = img.resize((94, 24))
        img = np.array(img)
        # 更换维度
        img = plate_dict_test.transform(img)
        # 添加维度(1,3,24,94)
        img = np.expand_dims(img, 0)
        # 车牌识别
        loc = plate_dict_ort_session.run(None, {'input': img})
        label = plate_dict_test.predict(loc)
        lb = ""
        for i in label:
            lb += plate_dict_test.CHARS[i]

        plates.append(lb)
    return plates


if __name__ == '__main__':
    app.debug = False  # 设置调试模式，生产模式的时候要关掉debug
    app.run(host='0.0.0.0', port=5001, threaded=True)
