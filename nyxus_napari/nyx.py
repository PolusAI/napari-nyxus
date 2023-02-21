from typing import Union
from qtpy.QtWidgets import QWidget, QScrollArea, QTableWidget, QVBoxLayout,QTableWidgetItem
from qtpy import QtCore, QtGui, QtWidgets, uic
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


current_label = 1
labels = None
labels_added = False

class FeaturesWidget(QWidget):
    
    @QtCore.Slot(QtWidgets.QTableWidgetItem)
    def onClicked(self, it):
        print('clicked')
        state = not it.data(SelectedRole)
        it.setData(SelectedRole, state)
        it.setBackground(
            QtGui.QColor(100, 100, 100) if state else QtGui.QColor(0, 0, 0)
        )

@magic_factory
def widget_factory(
    viewer: napari.Viewer,
    Intensity: Image, 
    Segmentation: Image,
    Features: Features,
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
    import nyxus
    import os
    
    intensity_path = str(Intensity.source.path)
    segmentation_path = str(Segmentation.source.path)
    
    
    #print(str(Intensity))
    
    nyxus_object = None
    
    if (Use_CUDA_Enabled_GPU):
        import subprocess
        
        try:
            subprocess.check_output('nvidia-smi')
            show_info('Nvidia GPU detected')
        except Exception: # this command not being found can raise quite a few different errors depending on the configuration
            show_info('No Nvidia GPU found. The machine must have a CUDA enable Nvidia GPU with drivers installed.')
            return
            
        nyxus_object = nyxus.Nyxus([Features.value], 
                               neighbor_distance=Neighbor_distance, 
                               pixels_per_micron=Pixels_per_micron, 
                               coarse_gray_depth=Coarse_gray_depth,
                               #n_feature_calc_threads=Number_of_calculation_threads,
                               #n_loader_threads=Number_of_loader_threads,
                               using_gpu = GPU_id)
        
    else:
        nyxus_object = nyxus.Nyxus([Features.value], 
                                neighbor_distance=Neighbor_distance, 
                                pixels_per_micron=Pixels_per_micron, 
                                coarse_gray_depth=Coarse_gray_depth,
                                #n_feature_calc_threads=Number_of_calculation_threads,
                                #n_loader_threads=Number_of_loader_threads,
                                using_gpu = -1)
    
    #result = nyxus_object.featurize_memory(Intensity.data, Segmentation.data)
  
    if (not os.path.isfile(segmentation_path) and not os.path.isdir(segmentation_path)):
        
        #save and load image data until in memory api is complete
        from PIL import Image
        
        #if (not Segmentation.data):
        #    show_info("Invalid segmentation input")
        #    return
            
        im = Image.fromarray(Segmentation.data)
        im.save('segmentation.tif')
        segmentation_path = 'segmentation.tif'
        
    if (not os.path.isfile(intensity_path) and not os.path.isdir(intensity_path)):
        
        #save and load image data until in memory api is complete
        from PIL import Image
        
        if (not Segmentation.data):
            show_info("Invalid intensity input")
            return
            
        im = Image.fromarray(Segmentation.data)
        im.save('intensity.tif')
        segmentation_path = 'intensity.tif'
    

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


    if (Save_to_csv):
        show_info("Saving results to " + Output_path + "out.csv")
        result.to_csv(Output_path + 'out.csv', sep='\t', encoding='utf-8')
    
    # Create window for the DataFrame viewer
    win = FeaturesWidget()
    scroll = QScrollArea()
    layout = QVBoxLayout()
    table = QTableWidget()
    scroll.setWidget(table)
    layout.addWidget(table)
    win.setLayout(layout)    
    win.setWindowTitle("Feature Results")

    # Add DataFrame to widget window
    table.setColumnCount(len(result.columns))
    table.setRowCount(len(result.index))
    table.setHorizontalHeaderLabels(result.columns)
    for i in range(len(result.index)):
        for j in range(len(result.columns)):
            table.setItem(i,j,QTableWidgetItem(str(result.iloc[i, j])))

        
    global labels
    global current_label
    
    seg = Segmentation.data
    labels = np.zeros_like(seg)
    def highlight_value(value):
        
        global current_label
        global labels
        global labels_added

        removed = False

        for ix, iy in np.ndindex(seg.shape):

            if (int(seg[ix, iy]) == int(value)):

                if (labels[ix, iy] != 0):
                    labels[ix, iy] = 0
                else:
                    labels[ix, iy] = int(value)
        
        if (not removed):
            current_label += 1
            
        if (not labels_added):
            viewer.add_labels(np.array(labels).astype('int8'), name="Selected ROI")
            labels_added = True
        else:
            viewer.layers["Selected ROI"].data = np.array(labels).astype('int8')

            
    def cell_was_clicked(self, event):
        current_column = table.currentColumn()
        
        if(current_column == 2):
            current_row = table.currentRow()
            cell_value = table.item(current_row, current_column).text()
            
            highlight_value(cell_value)
       
    table.cellClicked.connect(cell_was_clicked)

    # add DataFrame to Viewer
    viewer.window.add_dock_widget(win)
    
    #layer = viewer.layers[str(Segmentation)]
    #print(viewer.layers)
    
    @Segmentation.mouse_drag_callbacks.append
    def clicked_roi(layer, event):
        coords = np.round(event.position).astype(int)
        value = layer.data[coords[0]][coords[1]]
        table.selectRow(value)
    