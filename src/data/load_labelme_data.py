# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2020/3/12

from src.model.text_box import RectangleTextBox, QuadTextBox


def com_load_shape_data(shape_data):
    return RectangleTextBox.load_shape_data(shape_data) \
            if shape_data["shape_type"] == "rectangle" else \
            QuadTextBox.load_shape_data(shape_data)
