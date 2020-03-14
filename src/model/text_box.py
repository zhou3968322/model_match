# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2020/3/12
import numpy as np
import cv2
from numpy.linalg import norm

"""
QuadrilateralTextBox
"""


class TextBox(object):
    
    def __init__(self, points_array, label="", shape_type="polygon"):
        if isinstance(points_array, list):
            self._pt_arr = np.array(points_array, dtype=np.float32)
        else:
            self._pt_arr = points_array.astype(np.float32)
        assert isinstance(self._pt_arr, np.ndarray) and self._pt_arr.shape[1] == 2
        self.label = label
        self.shape_type = shape_type
        assert self.xmax != self.xmin and self.ymax != self.ymin

    @property
    def xmax(self):
        return np.max(self._pt_arr[:, 0])

    @property
    def xmin(self):
        return np.min(self._pt_arr[:, 0])

    @property
    def ymax(self):
        return np.max(self._pt_arr[:, 1])

    @property
    def ymin(self):
        return np.min(self._pt_arr[:, 1])
    
    @property
    def center(self):
        return np.mean(self._pt_arr, axis=0)


class QuadTextBox(TextBox):
    """
    目前文本框都是四点的结合，并且附带点和形状属性,顺时针点的集合,凸四边形
    :param pt0:
    :param pt1:
    :param pt2:
    :param pt3:
    :param text: 文本内容
    :param type:
    """
    
    def __init__(self, pt0, pt1, pt2, pt3, label="", shape_type="quad"):
        super(QuadTextBox, self).__init__([pt0, pt1, pt2, pt3], label, shape_type)

    @property
    def dr_mas(self):
        """Return 长的方向是x方向还是y"""
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin
        return "x" if width >= height else "y"

    @property
    def mas_points(self):
        """return 长边上的点对"""
        if self.dr_mas == "x":
            for i, pt in enumerate(self._pt_arr):
                if pt[0] == self.xmin:
                    n_p = self._pt_arr[(i + 1) % 4]
                    n_n_p = self._pt_arr[(i + 2) % 4]
                    n_n_n_p = self._pt_arr[(i + 3) % 4]
                    if n_n_n_p[0] > n_p[0]:
                        return np.array([[n_p, n_n_p], [pt, n_n_n_p]])
                    else:
                        return np.array([pt, n_p], [n_n_n_p, n_n_p])
                elif pt[0] == self.xmax:
                    n_p = self._pt_arr[(i + 1) % 4]
                    n_n_p = self._pt_arr[(i + 2) % 4]
                    n_n_n_p = self._pt_arr[(i + 3) % 4]
                    if n_n_n_p[0] > n_p[0]:
                        return np.array([[n_n_n_p, pt], [n_n_p, n_p]])
                    else:
                        return np.array([[n_p, pt], [n_n_p, n_n_n_p]])
        else:
            for i, pt in enumerate(self._pt_arr):
                if pt[1] == self.ymin:
                    n_p = self._pt_arr[(i + 1) % 4]
                    n_n_p = self._pt_arr[(i + 2) % 4]
                    n_n_n_p = self._pt_arr[(i + 3) % 4]
                    if n_n_n_p[1] < n_p[1]:
                        return np.array([[n_n_n_p, n_n_p], [pt, n_p]])
                    else:
                        return np.array([[pt, n_n_n_p], [n_p, n_n_p]])
                elif pt[1] == self.ymax:
                    n_p = self._pt_arr[(i + 1) % 4]
                    n_n_p = self._pt_arr[(i + 2) % 4]
                    n_n_n_p = self._pt_arr[(i + 3) % 4]
                    if n_n_n_p[1] < n_p[1]:
                        return np.array([[n_n_p, n_p], [n_n_n_p, pt]])
                    else:
                        return np.array([[n_p, pt], [n_n_p, n_n_n_p]])

    @property
    def side_ar(self):
        return np.array([norm(self._pt_arr[i] - self._pt_arr[(i + 1) % 4]) for i in range(4)], dtype=np.float32)
        
    @property
    def aspect_ratio(self):
        side1_dis = self.side_ar[0] + self.side_ar[2]
        side2_dis = self.side_ar[1] + self.side_ar[3]
        return max(side1_dis, side2_dis) / min(side2_dis, side1_dis)

    def draw_in(self, img):
        cv2.polylines(img, np.array([self._pt_arr.astype(np.int32)]), isClosed=True, color=(0, 0, 255), thickness=2)

    def get_relation(self, n_text_box):
        """只有都为x方向的文本框才有relation,
            如果都水平，则n_text_box在text_box下方为1，上方为-1，其他情况为0；
            如果都竖直，则n_text_box在text_box右方为1，左方为-1，其他情况为0"""
        if self.dr_mas != n_text_box.dr_mas:
            return 0
        if self.dr_mas == "x":
            if n_text_box.ymin >= self.ymax:
                return 1
            elif self.ymin >= n_text_box.ymax:
                return -1
            else:
                return 0
        else:
            if n_text_box.xmin >= self.xmax:
                return 1
            elif self.xmin >= n_text_box.xmax:
                return -1
            else:
                return 0
        # if self.dr_mas == "x":
        #     if n_text_box.center[1] > self.center[1]:
        #         return 1
        #     elif self.center[1] > n_text_box.center[1]:
        #         return -1
        #     else:
        #         return 0
        # else:
        #     if n_text_box.center[0] > self.center[0]:
        #         return 1
        #     elif self.center[0] > n_text_box.center[0]:
        #         return -1
        #     else:
        #         return 0
    
    @classmethod
    def load_shape_data(cls, shape_data):
        assert shape_data["shape_type"] == "polygon" and len(shape_data["points"]) == 4
        return cls(shape_data["points"][0], shape_data["points"][1],
                   shape_data["points"][2], shape_data["points"][3],
                   shape_data["label"])


class RectangleTextBox(QuadTextBox):
    
    def __init__(self, pt0, pt1, label="", shape_type="rectangle"):
        """
        pt0, pt1 分别为矩形左上角和右下角的点，或者矩形左下角和右上角的点
        :param pt0:
        :param pt1:
        :param label:
        :param shape_type:
        """
        xmin = min(pt0[0], pt1[0])
        xmax = max(pt0[0], pt1[0])
        ymin = min(pt0[1], pt1[1])
        ymax = max(pt0[1], pt1[1])
        super(RectangleTextBox, self).__init__([xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax],
                                               label, shape_type)

    @property
    def aspect_ratio(self):
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin
        return max(width, height) / min(width, height)

    @property
    def mas_points(self):
        if self.dr_mas == "x":
            return np.array([[[self.xmin, self.ymin], [self.xmax, self.ymin]],
                             [[self.xmin, self.ymax], [self.xmax, self.ymax]]])
        else:
            return np.array([[[self.xmin, self.ymin], [self.xmin, self.ymax]],
                             [[self.xmax, self.ymin], [self.xmax, self.ymax]]])
        

    @classmethod
    def load_shape_data(cls, shape_data):
        assert shape_data["shape_type"] == "rectangle"
        return cls(shape_data["points"][0], shape_data["points"][1], shape_data["label"], shape_data["shape_type"])


