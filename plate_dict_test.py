import numpy as np

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]




def transform(img):
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))  # shape= (z,x,y)  也就是（3 通道，24 height，94 width）

    return img


def predict(prebs):
    # prebs = torch.from_numpy(prebs[0])
    # greedy decode
    prebs = np.asarray(prebs[0])

    preb_labels = list()
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)

    ### label 就是预测的结果
    for i, label in enumerate(preb_labels):
        # if len(label) > 8:
        #     label = label[:8]
        # show image and its predict label
        return label


# image = cv2.imread("00_guiA40688.jpg")
# ## 更换维度
# img = transform(image)
#
#
#
# ## 增加一个维度(1,3,24,94)
# img = np.expand_dims(img, 0)
#
# print(img.shape)
#
# model_path = "lprnet.onnx"
# ort_session = ort.InferenceSession(model_path)
#
# startTime = time.time()
# loc = ort_session.run(None, {'input': img})
# l = predict(loc)
# lb = ""
# for i in l:
#     lb += CHARS[i]
# print(lb)
# endTime = time.time()
# print("未量化模型用时：", str(endTime - startTime) + "秒")
