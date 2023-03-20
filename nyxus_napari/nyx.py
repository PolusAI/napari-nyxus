from typing import Union
from qtpy.QtWidgets import QWidget, QScrollArea, QTableWidget, QVBoxLayout,QTableWidgetItem
from qtpy import QtCore, QtGui, QtWidgets, uic
import napari
from napari.layers import Image
from napari.utils.notifications import show_info

from magicgui import magic_factory

from nyxus_napari import nyx_napari
from enum import Enum

class Features(Enum):
    All = "*ALL*"
    Intensity = "*ALL_INTENSITY*"
    All_Morphology = "*ALL_MORPHOLOGY*"
    Basic_Morphology = "*BASIC_MORPHOLOGY*"
    GLCM = "*ALL_GLCM*"
    GLRM = "*ALL_GLRM*"
    GLSZM = "*ALL_GLSZM*"
    GLDM = "*ALL_GLDM*"
    NGTDM = "*ALL_NGTDM*"
    All_but_Gabor = "*ALL_BUT_GABOR*"
    All_but_GLCM= "*ALL_BUT_GLCM*"

Widget = Union["magicgui.widgets.Widget", "qtpy.QtWidgets.QWidget"]


current_label = 1
labels = None
labels_added = False

colormap = None
colormap_added = False


@magic_factory
def widget_factory(
    viewer: napari.Viewer,
    Intensity: Image, 
    Segmentation: Image,
    Features: nyx_napari.Features,
    Save_to_csv: bool = True,
    Output_path: "str" = "",
    Neighbor_distance: float = 5.0,
    Pixels_per_micron: float = 1.0,
    Coarse_gray_depth: int = 256, 
    Use_CUDA_Enabled_GPU: bool = False,
    GPU_id: int = 0):
    
    #wait for function call to load large modules
    import pandas as pd
    import numpy as np
    import os
    
    nyx = nyx_napari.NyxusNapari(
        viewer,
        Intensity, 
        Segmentation,
        Features,
        Save_to_csv,
        Output_path,
        Neighbor_distance,
        Pixels_per_micron,
        Coarse_gray_depth, 
        Use_CUDA_Enabled_GPU,
        GPU_id)
    
    nyx.calculate()

    nyx.add_features_table()
    
    
    
    