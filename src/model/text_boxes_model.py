# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2020/3/12
import numpy as np
import cv2
from src.data.load_labelme_data import com_load_shape_data
from src.common.utils import base64_cv2
from scipy.spatial.distance import cdist
from conf.conf import MIN_ASPECT_RATIO, MAX_ASPECT_RATIO, ASPECT_RATIO_ERROR


class TextBoxesBase(object):
    
    def __init__(self, im, boxes):
        self.boxes = boxes
        self._im = im
        self._img_height = im.shape[0]
        self._img_width = im.shape[1]
        self._img_size = (im.shape[1], im.shape[0])
        
    @property
    def im(self):
        return self._im
        
    @property
    def img_size(self):
        return self._img_size
        
    @classmethod
    def load_labelme_data(cls, labelme_data):
        b64_string = labelme_data["imageData"]
        im = base64_cv2(b64_string)
        boxes = [com_load_shape_data(shape_data) for shape_data in labelme_data["shapes"]]
        return cls(im, boxes)

    @property
    def ar_np2(self):
        return np.array([[text_box.aspect_ratio, 0] for text_box in self.boxes])

    @property
    def ar_np(self):
        return np.array([text_box.aspect_ratio for text_box in self.boxes])

    def get_box_direction(self, box):
        box_relation_np = np.array([box.get_relation(base_box) for base_box in self.boxes])
        box_relation_sum1 = np.sum(box_relation_np == 1)
        box_relation_sum0 = np.sum(box_relation_np == 0)
        box_relation_sum_1 = np.sum(box_relation_np == -1)
        if (box_relation_sum1 - box_relation_sum_1)  > 0.5 * (len(self.boxes) - box_relation_sum0):
            return 1
        elif (box_relation_sum_1 - box_relation_sum1)  > 0.5 * (len(self.boxes) - box_relation_sum0):
            return -1
        return 0

    def get_i_relation(self, i):
        """
        获取boxes中的第i个box相对于其他box的方向
        :param i:
        :return:
        """
        return self.get_box_direction(self.boxes[i])


def _get_affine_matrix(model_box, sample_box, mb_direction, sb_direction):
    src_points = sample_box.mas_points.reshape(4, 2)
    if mb_direction == 0 or sb_direction == 0:
        dst_points1 = model_box.mas_points.reshape(4, 2)
        dst_points2 = model_box.mas_points[::-1].reshape(4, 2)
        tf_matrix1 = cv2.getPerspectiveTransform(src_points, dst_points1)
        tf_matrix2 = cv2.getPerspectiveTransform(src_points, dst_points2)
        return [tf_matrix1, tf_matrix2]
    elif mb_direction == sb_direction:
        dst_points = model_box.mas_points.reshape(4, 2)
        tf_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return [tf_matrix]
    else:
        dst_points = model_box.mas_points[::-1].reshape(4, 2)
        tf_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return [tf_matrix]


class TextBoxesModel(TextBoxesBase):
    
    def __init__(self, limit_min_ar, limit_max_ar, ar_erro, im, boxes):
        super(TextBoxesModel, self).__init__(im, boxes)
        self._limit_min_ar = limit_min_ar
        self._limit_max_ar = limit_max_ar
        self._ar_error = ar_erro
        self._sort_index = np.argsort(self.ar_np2[:, 0])

    def filter(self, text_boxes_base):
        new_text_boxes = [text_box for text_box in text_boxes_base.boxes \
                          if self._limit_min_ar <= text_box.aspect_ratio < self._limit_max_ar]
        return TextBoxesBase(text_boxes_base.im, new_text_boxes)

    def get_ratio_dis(self, text_boxes_sample):
        ratio_matrix = cdist(self.ar_np2, text_boxes_sample.ar_np2)
        return ratio_matrix
        
    def match_affine_transform(self, text_boxes_sample):
        filter_boxes_sample = self.filter(text_boxes_sample)
        ratio_matrix = self.get_ratio_dis(filter_boxes_sample)
        i = len(self._sort_index) - 1
        while i >= 0:
            ratio_dis = ratio_matrix[self._sort_index[i], :]
            sample_indexes = np.where(abs(ratio_dis) <= self._ar_error)[0]
            model_box = self.boxes[self._sort_index[i]]
            if len(sample_indexes) == 1:
                sample_box = filter_boxes_sample.boxes[sample_indexes[0]]
                mb_direction = self.get_i_relation(self._sort_index[i])
                sb_direction = filter_boxes_sample.get_i_relation(sample_indexes[0])
                tf_matrixs = _get_affine_matrix(model_box, sample_box, mb_direction, sb_direction)
                for tf_matrix in tf_matrixs:
                    res_im = cv2.warpPerspective(filter_boxes_sample.im, tf_matrix, self.img_size)
                    return res_im
                break
            elif len(sample_indexes) >= 2:
                sample_box = filter_boxes_sample.boxes[sample_indexes[1]]
                mb_direction = self.get_i_relation(self._sort_index[i])
                sb_direction = filter_boxes_sample.get_i_relation(sample_indexes[1])
                tf_matrixs = _get_affine_matrix(model_box, sample_box, mb_direction, sb_direction)
                for tf_matrix in tf_matrixs:
                    res_im = cv2.warpPerspective(filter_boxes_sample.im, tf_matrix, self.img_size)
                    return res_im
                break
            i -= 1
        print("debugging")

    @classmethod
    def load_labelme_data(cls, labelme_data, **kwargs):
        b64_string = labelme_data["imageData"]
        im = base64_cv2(b64_string)
        boxes = [com_load_shape_data(shape_data) for shape_data in labelme_data["shapes"]]
        min_ar = kwargs.get("min_ar", MIN_ASPECT_RATIO)
        max_ar = kwargs.get("max_ar", MAX_ASPECT_RATIO)
        ar_error = kwargs.get("ar_error", ASPECT_RATIO_ERROR)
        return cls(min_ar, max_ar, ar_error, im, boxes)
