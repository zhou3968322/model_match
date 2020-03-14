# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2020/3/12

import sys
import os
PROJECT_ROOT_PATH = os.path.abspath(os.path.join(__file__, '../../'))
print(PROJECT_ROOT_PATH)
sys.path.append(PROJECT_ROOT_PATH)

import unittest, json, codecs, cv2
from src.model.text_boxes_model import TextBoxesModel, TextBoxesBase


class TestModelMatcher(unittest.TestCase):
    
    def setUp(self):
        model_data_path = os.path.join(PROJECT_ROOT_PATH, 'tests/testdata/train_invoice.json')
        with codecs.open(model_data_path, 'r', 'utf-8') as fr:
            self.model_data = json.loads(fr.read())
    
    def test_model_cluster_match1(self):
        sample_data_path = os.path.join(PROJECT_ROOT_PATH, 'tests/testdata/sample1.json')
        res_img_path = os.path.join(PROJECT_ROOT_PATH, 'tests/testdata/sample1_res.jpg')
        with codecs.open(sample_data_path, 'r', 'utf-8') as fr:
            sample_data = json.loads(fr.read())
        text_boxes_model = TextBoxesModel.load_labelme_data(self.model_data)
        text_boxes_sample = TextBoxesBase.load_labelme_data(sample_data)
        res_im = text_boxes_model.match_affine_transform(text_boxes_sample)
        cv2.imwrite(res_img_path, res_im)

    def test_model_cluster_match2(self):
        sample_data_path = os.path.join(PROJECT_ROOT_PATH, 'tests/testdata/sample2.json')
        res_img_path = os.path.join(PROJECT_ROOT_PATH, 'tests/testdata/sample2_res.jpg')
        with codecs.open(sample_data_path, 'r', 'utf-8') as fr:
            sample_data = json.loads(fr.read())
        text_boxes_model = TextBoxesModel.load_labelme_data(self.model_data)
        text_boxes_sample = TextBoxesBase.load_labelme_data(sample_data)
        res_im = text_boxes_model.match_affine_transform(text_boxes_sample)
        cv2.imwrite(res_img_path, res_im)


if __name__ == '__main__':
    unittest.main()
