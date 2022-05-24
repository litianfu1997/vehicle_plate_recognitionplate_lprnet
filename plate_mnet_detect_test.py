import warnings
import argparse
import numpy as np
from config import cfg_mnet
from itertools import product as product
from math import ceil

parser = argparse.ArgumentParser(description='RetinaPL')

parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=1000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

args = parser.parse_args()

cfg = cfg_mnet


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
                        # print(anchors)
                        # exit(1)

        output = np.asarray(anchors)

        output = np.resize(output, (int(output.shape[0] / 4), 4))

        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = np.concatenate((
        np.add(priors[:, :2], np.multiply(np.multiply(loc[:, :2], variances[0]), priors[:, 2:])),
        np.multiply(priors[:, 2:], np.exp(np.multiply(loc[:, 2:], variances[1])))), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = np.concatenate((np.add(priors[:, :2], np.multiply(np.multiply(pre[:, :2], variances[0]), priors[:, 2:])),
                             np.add(priors[:, :2], np.multiply(np.multiply(pre[:, 2:4], variances[0]), priors[:, 2:])),
                             np.add(priors[:, :2], np.multiply(np.multiply(pre[:, 4:6], variances[0]), priors[:, 2:])),
                             np.add(priors[:, :2], np.multiply(np.multiply(pre[:, 6:8], variances[0]), priors[:, 2:])),
                             ), 1)
    return landms


def get_plate_pos(img_raw, ort_session, resize=1):
    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = np.asarray([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=float)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loc, conf, landms = ort_session.run(None, {'input': img})

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    prior_data = priors
    boxes = decode(loc.squeeze(0), prior_data, cfg['variance'])

    boxes = boxes * scale / resize

    scores = conf.squeeze(0)[:, 1]

    landms = decode_landm(landms.squeeze(0), prior_data, cfg['variance'])

    scale1 = np.asarray([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                         img.shape[3], img.shape[2],
                         img.shape[3], img.shape[2]])

    landms = landms * scale1 / resize

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]
    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]
    dets = np.concatenate((dets, landms), axis=1)
    # show image
    bboxes = []
    for b in dets:
        if b[4] < args.vis_thres:
            continue
        b = list(map(int, b))
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        bboxes.append((x1, y1, x2, y2))
        return bboxes
