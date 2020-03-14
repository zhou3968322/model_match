# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2020/3/12


import sys
import os

PROJECT_ROOT_PATH = os.path.abspath(os.path.join(__file__, '../../'))
print(PROJECT_ROOT_PATH)
sys.path.append(PROJECT_ROOT_PATH)

import unittest, json, codecs, cv2
import numpy as np
from src.model.text_boxes_model import TextBoxesBase


class TestLoadLabelMeData(unittest.TestCase):
    
    def test_load_model_data(self):
        data_path = os.path.join(PROJECT_ROOT_PATH, 'tests/testdata/train_invoice.json')
        with codecs.open(data_path, 'r', 'utf-8') as fr:
            label_data = json.loads(fr.read())
        text_boxes_model = TextBoxesBase.load_labelme_data(label_data)
        debug_path = os.path.join(PROJECT_ROOT_PATH, 'tests/testdata/train_invoice_debug.jpg')
        debug_im = text_boxes_model.im
        for box in text_boxes_model.boxes:
            box.draw_in(debug_im)
        cv2.imwrite(debug_path, debug_im)
        max_index = np.argmax(text_boxes_model.ar_np2[:, 0])
        min_index = np.argmin(text_boxes_model.ar_np2[:, 0])
        print("get ratios")
        print("max_ratio:{},text:{}".format(text_boxes_model.ar_np2[:, 0][max_index], text_boxes_model.boxes[max_index].label))
        print("min_ratio:{}:text:{}".format(text_boxes_model.ar_np2[:, 0][min_index], text_boxes_model.boxes[min_index].label))

    def test_load_sample_data1(self):
        data_path = os.path.join(PROJECT_ROOT_PATH, 'tests/testdata/sample1.json')
        with codecs.open(data_path, 'r', 'utf-8') as fr:
            label_data = json.loads(fr.read())
        text_boxes_model = TextBoxesBase.load_labelme_data(label_data)
        debug_path = os.path.join(PROJECT_ROOT_PATH, 'tests/testdata/sample1_debug.jpg')
        debug_im = text_boxes_model.im
        for box in text_boxes_model.boxes:
            box.draw_in(debug_im)
        cv2.imwrite(debug_path, debug_im)
        max_index = np.argmax(text_boxes_model.ar_np2[:, 0])
        min_index = np.argmin(text_boxes_model.ar_np2[:, 0])
        print("get ratios")
        print("max_ratio:{},text:{}".format(text_boxes_model.ar_np2[:, 0][max_index],
                                            text_boxes_model.boxes[max_index].label))
        print("min_ratio:{}:text:{}".format(text_boxes_model.ar_np2[:, 0][min_index],
                                            text_boxes_model.boxes[min_index].label))

    def test_load_sample_data2(self):
        data_path = os.path.join(PROJECT_ROOT_PATH, 'tests/testdata/sample2.json')
        with codecs.open(data_path, 'r', 'utf-8') as fr:
            label_data = json.loads(fr.read())
        text_boxes_model = TextBoxesBase.load_labelme_data(label_data)
        debug_path = os.path.join(PROJECT_ROOT_PATH, 'tests/testdata/sample2_debug.jpg')
        debug_im = text_boxes_model.im
        for box in text_boxes_model.boxes:
            box.draw_in(debug_im)
        cv2.imwrite(debug_path, debug_im)
        max_index = np.argmax(text_boxes_model.ar_np2[:, 0])
        min_index = np.argmin(text_boxes_model.ar_np2[:, 0])
        print("get ratios")
        print("max_ratio:{},text:{}".format(text_boxes_model.ar_np2[:, 0][max_index],
                                            text_boxes_model.boxes[max_index].label))
        print("min_ratio:{}:text:{}".format(text_boxes_model.ar_np2[:, 0][min_index],
                                            text_boxes_model.boxes[min_index].label))


if __name__ == '__main__':
    unittest.main()
