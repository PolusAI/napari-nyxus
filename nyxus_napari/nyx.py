
# example_plugin.some_module
#from types import UnionType
from typing import Union
from qtpy.QtWidgets import QWidget
import napari
from napari.layers import Image
from napari.utils.notifications import show_info

from magicgui import magic_factory

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

        

@magic_factory
def widget_factory(
    viewer: napari.Viewer,
    Intensity: Image, 
    Segmentation: Image,
    Features: Features,
    Output_path: "str" = "",
    Neighbor_distance: float = 5.0,
    Pixels_per_micron: float = 1.0,
    Coarse_gray_depth: int = 256, 
    Number_of_calculation_threads: int = 4,
    Number_of_loader_threads: int = 1,
    Use_GPU: bool = False,
    GPU_id: int = 0):
    
    #wait for function call to load large modules
    import pandas as pd
    import nyxus
    import os
    from qtpy.QtWidgets import QWidget,QScrollArea, QTableWidget, QVBoxLayout,QTableWidgetItem
    
    intensity_path = str(Intensity.source.path)
    segmentation_path = str(Segmentation.source.path)
    
    nyxus_object = None
    
    if (Use_GPU):
        nyxus_object = nyxus.Nyxus([Features.value], 
                               neighbor_distance=Neighbor_distance, 
                               pixels_per_micron=Pixels_per_micron, 
                               coarse_gray_depth=Coarse_gray_depth,
                               n_feature_calc_threads=Number_of_calculation_threads,
                               n_loader_threads=Number_of_loader_threads,
                               using_gpu = GPU_id)
        
    else:
        nyxus_object = nyxus.Nyxus([Features.value], 
                                neighbor_distance=Neighbor_distance, 
                                pixels_per_micron=Pixels_per_micron, 
                                coarse_gray_depth=Coarse_gray_depth,
                                n_feature_calc_threads=Number_of_calculation_threads,
                                n_loader_threads=Number_of_loader_threads,
                                using_gpu = -1)

    result = None
    if (os.path.isdir(intensity_path)):
        if (not os.path.isdir(segmentation_path)):
            #throw error since both must be directories
            show_info("Intensity and Segmentation must both be a directory or both be a file.")
            return 
        
        
        filepattern = ".*"
        
        show_info("Calculating features...")
        result = nyxus_object.featurize_directory(intensity_path, segmentation_path, filepattern)
    
    elif (os.path.isfile(intensity_path)):
        if (not os.path.isfile(segmentation_path)):
            show_info("Intensity and Segmentation must both be a directory or both be a file.")
            return
        
        show_info("Calculating features...")
        result = nyxus_object.featurize([intensity_path], [segmentation_path])
    
    else:
       show_info("Invalid input type. Please load an image or directory of images.")
        
                    
    result.to_csv(Output_path + 'out.csv', sep='\t', encoding='utf-8')
    
    show_info("Saving results to " + Output_path + "out.csv")
    
    win = QWidget()
    scroll = QScrollArea()
    layout = QVBoxLayout()
    table = QTableWidget()
    scroll.setWidget(table)
    layout.addWidget(table)
    win.setLayout(layout)    
    win.setWindowTitle("Feature Results")

    table.setColumnCount(len(result.columns))
    table.setRowCount(len(result.index))
    table.setHorizontalHeaderLabels(result.columns)
    for i in range(len(result.index)):
        for j in range(len(result.columns)):
            table.setItem(i,j,QTableWidgetItem(str(result.iloc[i, j])))


    
    viewer.window.add_dock_widget(win)
    
    
    #mgui_table = Table(value=result)

    #widget.layout().addWidget(mgui_table.native)
    
    #return track
    
    
        
            
    
    """
    if (filepattern == ""):
        filepattern = ".*"
    
    nyxus_object = nyxus.Nyxus(['*ALL*'])
    
    result = nyxus_object.featurize_directory(Intensity, Segmentation, filepattern)
    
    result.to_csv(path + 'out.csv', sep='\t', encoding='utf-8')
    
    return 0
    """





"""
from napari.utils.notifications import show_info
import napari.viewer
import napari
from napari.utils import nbscreenshot
import pprint

def show_hello_message() :
    
    viewer = napari.Viewer()
    
    data = viewer.layers
    
    for layer in data:
        pprint.pprint(layer)
    
    #pprint.pprint(data)

    
def get_data(image):
    print("success")
"""

"""
if (napari.viewer.layers[0].length == 0):
    show_info("No data loaded")
    
data = napari.viewer.layers[0].data

print(data)
if (data.size() > 0):
    print(data[0])
"""