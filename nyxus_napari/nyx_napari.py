from typing import Union
from qtpy.QtWidgets import QWidget, QScrollArea, QTableWidget, QVBoxLayout,QTableWidgetItem
from qtpy import QtCore, QtGui, QtWidgets, uic
import napari
from napari.layers import Image
from napari.utils.notifications import show_info
from magicgui import magic_factory
from enum import Enum
import numpy as np
import pandas as pd

import nyxus

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
    
class FeaturesWidget(QWidget):
    
    @QtCore.Slot(QtWidgets.QTableWidgetItem)
    def onClicked(self, it):
        print('clicked')
        state = not it.data(SelectedRole)
        it.setData(SelectedRole, state)
        it.setBackground(
            QtGui.QColor(100, 100, 100) if state else QtGui.QColor(0, 0, 0)
        )

class NyxusNapari:
    
    def __init__(
        self,
        viewer: napari.Viewer,
        intensity: Image, 
        segmentation: Image,
        features: Features,
        save_to_csv: bool = True,
        output_path: "str" = "",
        neighbor_distance: float = 5.0,
        pixels_per_micron: float = 1.0,
        coarse_gray_depth: int = 256, 
        use_CUDA_Enabled_GPU: bool = False,
        gpu_id: int = 0):
    
        self.viewer = viewer
        self.intensity = intensity
        self.segmentation = segmentation
        self.save_to_csv = save_to_csv
        self.output_path = output_path
    
        self.nyxus_object = None
        self.result = None

        self.current_label = 0
        self.seg = self.segmentation.data
        self.labels = np.zeros_like(self.seg)
        self.colormap = np.zeros_like(self.seg)
        self.colormap_added = False
        
        self.labels_added = False
        
        if (use_CUDA_Enabled_GPU):
            import subprocess
            
            try:
                subprocess.check_output('nvidia-smi')
                show_info('Nvidia GPU detected')
            except Exception: # this command not being found can raise quite a few different errors depending on the configuration
                show_info('No Nvidia GPU found. The machine must have a CUDA enable Nvidia GPU with drivers installed.')
                return
                
            self.nyxus_object = nyxus.Nyxus([features.value], 
                                neighbor_distance=neighbor_distance, 
                                pixels_per_micron=pixels_per_micron, 
                                coarse_gray_depth=coarse_gray_depth,
                                using_gpu = gpu_id)
            
        else:
            self.nyxus_object = nyxus.Nyxus([features.value], 
                                    neighbor_distance=neighbor_distance, 
                                    pixels_per_micron=pixels_per_micron, 
                                    coarse_gray_depth=coarse_gray_depth,
                                    using_gpu = -1)
        
        @segmentation.mouse_drag_callbacks.append
        def clicked_roi(layer, event):
            coords = np.round(event.position).astype(int)
            value = layer.data[coords[0]][coords[1]]
            if (value == 0):
                return
            self.table.selectRow(value)
            
        @intensity.mouse_drag_callbacks.append
        def clicked_roi(layer, event):
            coords = np.round(event.position).astype(int)
            value = segmentation.data[coords[0]][coords[1]]
            if (value == 0):
                return
            self.table.selectRow(value)
    
    
    def calculate(self):  
        self.result = self.nyxus_object.featurize(self.intensity.data, self.segmentation.data)
        if (self.save_to_csv):
            show_info("Saving results to " + self.output_path + "out.csv")
            self.result.to_csv(self.output_path + 'out.csv', sep='\t', encoding='utf-8')
   

    def add_features_table(self):
        # Create window for the DataFrame viewer
        self.win = FeaturesWidget()
        scroll = QScrollArea()
        layout = QVBoxLayout()
        self.table = QTableWidget()
        scroll.setWidget(self.table)
        layout.addWidget(self.table)
        self.win.setLayout(layout)    
        self.win.setWindowTitle("Feature Results")

        # Add DataFrame to widget window
        self.table.setColumnCount(len(self.result.columns))
        self.table.setRowCount(len(self.result.index))
        self.table.setHorizontalHeaderLabels(self.result.columns)
        for i in range(len(self.result.index)):
            for j in range(len(self.result.columns)):
                self.table.setItem(i,j,QTableWidgetItem(str(self.result.iloc[i, j])))
                
        self.table.cellClicked.connect(self.cell_was_clicked)
        self.table.horizontalHeader().sectionClicked.connect(self.onHeaderClicked)

        # add DataFrame to Viewer
        self.viewer.window.add_dock_widget(self.win)
    
    def highlight_value(self, value):

        removed = False

        for ix, iy in np.ndindex(self.seg.shape):

            if (int(self.seg[ix, iy]) == int(value)):

                if (self.labels[ix, iy] != 0):
                    self.labels[ix, iy] = 0
                else:
                    self.labels[ix, iy] = int(value)
        
        if (not removed):
            self.current_label += 1
            
        if (not self.labels_added):
            self.viewer.add_labels(np.array(self.labels).astype('int8'), name="Selected ROI")
            self.labels_added = True
        else:
            self.viewer.layers["Selected ROI"].data = np.array(self.labels).astype('int8')

            
    def cell_was_clicked(self, event):
        current_column = self.table.currentColumn()
        
        if(current_column == 2):
            current_row = self.table.currentRow()
            cell_value = self.table.item(current_row, current_column).text()
            
            self.highlight_value(cell_value)
    
    def onHeaderClicked(self, logicalIndex):
        self.create_feature_color_map(logicalIndex)
        
    def create_feature_color_map(self, logicalIndex):
        self.colormap = np.zeros_like(self.seg)
        
        labels = self.result.iloc[:,2]
        values = self.result.iloc[:,logicalIndex]
        label_values = pd.Series(values, index=labels).to_dict()
        
        for ix, iy in np.ndindex(self.seg.shape):
            
            if (self.seg[ix, iy] != 0):
                if(np.isnan(label_values[int(self.seg[ix, iy])])):
                    continue
                
                self.colormap[ix, iy] = label_values[int(self.seg[ix, iy])]
        
        if (not self.colormap_added):
            self.viewer.add_image(np.array(self.colormap), name="Colormap")
            self.colormap_added = True
        else:
            self.viewer.layers["Colormap"].data = np.array(self.colormap)
    