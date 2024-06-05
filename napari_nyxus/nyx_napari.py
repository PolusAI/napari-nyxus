from qtpy.QtWidgets import QWidget, QCheckBox, QScrollArea, QTableWidget, QVBoxLayout,QTableWidgetItem, QLineEdit, QLabel, QHBoxLayout, QPushButton, QComboBox
from qtpy.QtCore import Qt, QTimer
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtGui import QColor
from superqt import QLabeledDoubleRangeSlider
import napari
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from magicgui import magic_factory

from enum import Enum
import numpy as np
import pandas as pd
import dask
from filepattern import FilePattern
import tempfile

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from napari_nyxus.util import util
from napari_nyxus.util import rotated_header

#from napari_nyxus.table import TableWidget, add_table, get_table
from napari_skimage_regionprops import TableWidget, add_table, get_table
from napari_workflows._workflow import _get_layer_from_data

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
        state = not it.data(SelectedRole)
        it.setData(SelectedRole, state)
        it.setBackground(
            QtGui.QColor(100, 100, 100) if state else QtGui.QColor(0, 0, 0)
        )

class NyxusNapari:
    """ Class to create Napair plugin to run Nyxus

    This class takes in arguments from the Napari viewer and uses the args
    to construct click events, run nyxus, and add other elements to the viewer
    based on nyxus results
    
    """
    
    def __init__(
        self,
        viewer: napari.Viewer,
        intensity: Image, 
        segmentation: Labels,
        features: Features,
        output_path: "str" = "",
        neighbor_distance: int = 5,
        pixels_per_micron: float = 1.0,
        coarse_gray_depth: int = 256, 
        use_CUDA_Enabled_GPU: bool = False,
        gpu_id: int = 0):

        # Set class data
        self.viewer = viewer
        self.intensity = intensity
        self.segmentation = segmentation
        self.output_path = output_path
    
        self.nyxus_object = None
        self.result = None

        self.current_label = 0
        self.seg = self.segmentation.data
        self.labels = np.zeros_like(self.seg)
        self.colormap = np.zeros_like(self.seg)
        self.colormap_added = False
        self.slider_added = False
        
        self.batched = False
        
        self.labels_added = False
        
        self.num_annotations = 0

        # Check for CUDA enable GPU if requested
        if (use_CUDA_Enabled_GPU):
            import subprocess
            
            try:
                subprocess.check_output('nvidia-smi')
                show_info('Nvidia GPU detected')
            except Exception: # this command not being found can raise quite a few different errors depending on the configuration
                show_info('No Nvidia GPU found. The machine must have a CUDA enable Nvidia GPU with drivers installed.')
                return
            
            # Create GPU enabled nyxus object
            self.nyxus_object = nyxus.Nyxus([features.value], 
                                neighbor_distance=neighbor_distance, 
                                pixels_per_micron=pixels_per_micron, 
                                coarse_gray_depth=coarse_gray_depth,
                                using_gpu = gpu_id)
            
        else:
            # No GPU nyxus objet
            self.nyxus_object = nyxus.Nyxus([features.value], 
                                    neighbor_distance=neighbor_distance, 
                                    pixels_per_micron=pixels_per_micron, 
                                    coarse_gray_depth=coarse_gray_depth,
                                    using_gpu = -1)
        
        # Add event handler for ROI clicking feature
        
        @segmentation.mouse_drag_callbacks.append
        def _after_labels_clicked(layer, event):
            """ Adds click event to segmentation image so when ROIs are clicked in the viewer,
            the correct ROI is highlighted in the results table
            """
            coords = np.round(event.position).astype(int)
            
            try:
                value = layer.data[coords[0]][coords[1]]
            except:
                return
            
            if (value == 0):
                return

            self.table.selectRow(value-1)
            
            self.after_click(value)
            
        @intensity.mouse_drag_callbacks.append
        def clicked_roi(layer, event, event1):
            """ Adds click event to intensity image so when ROIs are clicked in the viewer,
            the correct ROI is highlighted in the results table
            """
            coords = np.round(event.position).astype(int)
            
            try:
                value = segmentation.data[coords[0]][coords[1]]
            except:
                return
            
            if (value == 0):
                return

            self.table.selectRow(value-1)

    
    def after_click(self, value):
        self.table.selectRow(value-1)

  
    def run(self):  
        """ Run Nyxus on data from napari viewer
        """
        show_info("Calculating features...")
        self._run_calculate()
        self.add_features_table()
        self.is_heatmap_added = False


    def _run_calculate(self):
        """ Call correct calculates features. Should not be called directly.
        Used calling _calculate from woker threads instead of directly.
        This method is called from run()
        """
        #worker = self._calculate()
        #worker.start()
        #worker.run()
        self._calculate()
    
    #@thread_worker
    def _calculate(self):
        """ Calculates the features using Nyxus 
        """
        if (type(self.intensity.data) == dask.array.core.Array):
            self.batched = True
            self._calculate_out_of_core()
        else:
            self.result = self.nyxus_object.featurize(self.intensity.data, self.segmentation.data, intensity_names=[self.intensity.name], label_names=[self.segmentation.name])
            
    
    def _calculate_out_of_core(self):
        """ Out of core calculations for when dataset size is larger than what Napari
        loads into memory
        """
        from os import walk
        
        # Get files from directory and skip hidden files
        filenames = [f for f in next(walk(self.intensity.source.path), (None, None, []))[2] if not f.startswith('.')]
        filenames.sort() # sort files to be in same order they appear in napari


        self.result = None
        names_index = 0

        for idx in np.ndindex(self.intensity.data.numblocks):

            num_files = len(self.intensity.data.blocks[idx])
            names = filenames[names_index:names_index + num_files]

            # set DataFrame value after first batch is processed
            if self.result is None:
                self.result = self.nyxus_object.featurize(
                    self.intensity.data.blocks[idx].compute(), 
                    self.segmentation.data.blocks[idx].compute(),
                    intensity_names=names,
                    label_names=names
                )
                
            else: # Concat to first batch results
                self.result = pd.concat([self.result, 
                                         self.nyxus_object.featurize(
                                            self.intensity.data.blocks[idx].compute(), 
                                            self.segmentation.data.blocks[idx].compute(),
                                            intensity_names=names,
                                            label_names=names)], 
                                         ignore_index=True)
            
            names_index += num_files

            
        #self.result = pd.concat(results, ignore_index=True)
    
    
    def add_features_table(self):
        """ Appends table consisting of results dataframe from the feature calculations to the
        Napari viewer
        """
        show_info("Creating features table")

        self._add_features_table()

    def add_feature_calculation_table_options(self):
        
        # create window for feature calculation table functionality
        win = FeaturesWidget()
        scroll = QScrollArea()
        layout = QVBoxLayout()
        widget_table = QWidget()
        widget_table.setLayout(layout)

        # add combobox for selecting heatmap type
        self.heatmap_combobox = QComboBox()
        self.heatmap_combobox.addItems(plt.colormaps()[:5])
        self.heatmap_combobox.addItem('gray')
        widget_table.layout().addWidget(self.heatmap_combobox)

        self.remove_number_checkbox = QCheckBox("Hide feature calculation values")
        widget_table.layout().addWidget(self.remove_number_checkbox)
        
        # button to create heatmap
        heatmap_button = QPushButton("Generate heatmap")
        heatmap_button.clicked.connect(self.generate_heatmap)
        
        widget_table.layout().addWidget(heatmap_button)

        # add text box for selecting column for extracting annotations
        self.column_box = QComboBox()
        self.column_box.addItems(['intensity_image', 'mask_image'])

        # text box for filepattern for extracting annotations
        self.filepattern_box = QLineEdit()
        self.filepattern_box.setPlaceholderText("Enter filepattern (ex: r{r:d+}_c{c:d+}.tif)")
        self.filepattern_box.textChanged.connect(self.check_annotations_input)

        # add text box for selecting which filepattern variable to extract annotation
        self.annotation_box = QLineEdit()
        self.annotation_box.setPlaceholderText("Enter annotation to extract (ex:r)")
        self.annotation_box.textChanged.connect(self.check_annotations_input)

        self.annotation_button = QPushButton("Extract annotation")
        self.annotation_button.clicked.connect(self.extract_annotation)

        # add text for selecting column(s) to sort by
        self.sort_by_box = QLineEdit()
        self.sort_by_box.setPlaceholderText("Enter column to sort rows by")
        self.sort_by_box.textChanged.connect(self.check_sort_input)

        # add button for sorting columns
        self.sort_button = QPushButton("Sort")
        self.sort_button.clicked.connect(self._sort)
        
        # add widgets for feature calculations table
        widget_table.layout().addWidget(self.column_box)
        widget_table.layout().addWidget(self.filepattern_box)
        widget_table.layout().addWidget(self.annotation_box)
        widget_table.layout().addWidget(self.annotation_button)
        widget_table.layout().addWidget(self.sort_by_box)
        widget_table.layout().addWidget(self.sort_button)
        
        scroll.setWidget(widget_table)
        win.setLayout(layout)    
        win.setWindowTitle("Feature Table Options")


        self.viewer.window.add_dock_widget(win)

    
    def _add_features_table(self):   
        """ Adds table to Napari viewer
        """ 

        labels_layer = _get_layer_from_data(self.viewer, self.seg)
        new_table = self.result.to_dict()
        
        flipped_dict = {}
        for key, value in new_table.items():
            values_list = list(value.values())
            flipped_dict[key] = values_list
            
        
        labels_layer.properties = flipped_dict
        
        add_table(labels_layer, self.viewer)
        
        widget_table = get_table(labels_layer, self.viewer)

        self.table = widget_table._view

        self.table.setHorizontalHeader(rotated_header.RotatedHeaderView(self.table))
        
        self.table.cellDoubleClicked.connect(self.cell_was_clicked)

        # remove label clicking event to use our own
        try:
            widget_table._layer.mouse_drag_callbacks.remove(widget_table._clicked_labels)
        except:
            print('No mouse drag event to remove')

        # add new label clicking event
        self.table.horizontalHeader().sectionDoubleClicked.connect(self.onHeaderClicked)

        self.add_feature_calculation_table_options()

    def check_sort_input(self):

        if self.sort_by_box.text():
            self.sort_button.setEnabled(True)
        else:
            self.sort_button.setEnabled(False)

    def check_annotations_input(self):

        if self.filepattern_box.text() and self.annotation_box.text():
            self.annotation_button.setEnabled(True)
        else:
            self.annotation_button.setEnabled(False)
    
    def _sort(self):

        sort_columns = self.sort_by_box.text().split()

        # check if columns are valid
        for column in sort_columns:
            if column not in self.result.columns.to_list():
                show_info(f'Column name \"{column}\" is not a valid column.')
                return

        if (len(sort_columns) == 1): # sort table in place if only one sorting column is passed
            sort_column_index = self.result.columns.get_loc(sort_columns[0])

            self.table.sortItems(sort_column_index)

        else: # sort datafame when multiple columns are passed
            
            #self.result.sort_values(by=sort_columns, ascending=[i % 2 == 0 for i in range(len(sort_columns))], inplace=True)
            self.result.sort_values(by=sort_columns, inplace=True)

            for row in range(self.result.shape[0]):
                for col in range(self.result.shape[1]):
                    self.table.setItem(row, col, QTableWidgetItem(str(self.result.iat[row, col])))

            if self.is_heatmap_added:
                self.generate_heatmap()


    def generate_heatmap(self):

        remove = self.remove_number_checkbox.isChecked()

        if remove:
            row_height = self.table.rowHeight(1)
            column_width = self.table.columnWidth(1)

            width = min(row_height, column_width)

            self.table.horizontalHeader().setDefaultSectionSize(width)
            self.table.verticalHeader().setDefaultSectionSize(width)


        for col in range(3 + self.num_annotations, self.result.shape[1]):

            # Get the column data
            column_data = self.result.iloc[:, col]
            
            # Normalize feature calculation values between 0 and 1 for this column
            normalized_values = (column_data - column_data.min()) / (column_data.max() - column_data.min())

            # Map normalized values to colors using a specified colormap
            colormap = plt.get_cmap(self.heatmap_combobox.currentText())
            colors = (colormap(normalized_values) * 255).astype(int)  # Multiply by 255 to convert to QColor range
            # Set background color for each item in the column
            
            for row in range(self.result.shape[0]): 

                if remove:
                    self.table.item(row, col).setText('') # remove feature calculation value from cell

                self.table.item(row, col).setBackground(QColor(colors[row][0], colors[row][1], colors[row][2], colors[row][3]))

            self.is_heatmap_added = True
    
    def extract_annotation(self, event):
        import os
        
        column_name = self.column_box.currentText()
        file_pattern = self.filepattern_box.text()
        annotation = self.annotation_box.text()
                
        # write filenames to txt file to feed into filepattern (todo: update filepattern to remove need for text file)
        # use temp directory to allow filepattern (another process) to open temp file on Windows
        with tempfile.TemporaryDirectory() as td:
            f_name = os.path.join(td, 'rows')
            try:
                row_values = self.result[column_name].to_list()
            except:
                show_info("Invalid column name")
                return
            with open(f_name, 'w') as fh:
                for row in row_values:
                        fh.write(f"{row}\n")
            
            fp = FilePattern(f_name, file_pattern)
                
            found_annotations = []
            for annotations, file in fp:
                if annotation not in annotations:
                    continue
                found_annotations.append(annotations[annotation])

        if (len(found_annotations) != len(row_values)):
            show_info('Error extracting annotations. Check that the filenames match the filepattern.')
            return

        annotations_position = 3

        try:
            self.result.insert(annotations_position, annotation, found_annotations, allow_duplicates=False)
        except:
            show_info("Error inserting annotations column. Check that the name of the annotations column is unique.")
            return

        self.table.insertColumn(annotations_position)

        self.table.setHorizontalHeaderItem(annotations_position, QTableWidgetItem(annotation))

        for row in range(self.table.rowCount()):
            self.table.setItem(row, annotations_position, QTableWidgetItem(str(found_annotations[row])))

        self.num_annotations += 1

    def cell_was_clicked(self, event):
        
        if (self.batched):
            show_info('Feature not enabled for batched processing')
            return
        
        current_column = self.table.currentColumn()
        
        if(current_column == 2):
            current_row = self.table.currentRow()
            cell_value = self.table.item(current_row, current_column).text()
            
            self.highlight_value(cell_value)
    
    def highlight_value(self, value):
        """ 
        """
        
        if (self.batched):
            show_info('Feature not enabled for batched processing')
            return

        removed = False

        for ix, iy in np.ndindex(self.seg.shape):

            if (int(self.seg[ix, iy]) == int(float(value))):

                if (self.labels[ix, iy] != 0):
                    self.labels[ix, iy] = 0
                else:
                    self.labels[ix, iy] = int(float(value))
        
        if (not removed):
            self.current_label += 1
            
        if (not self.labels_added):
            self.viewer.add_labels(np.array(self.labels).astype('int8'), name="Selected ROI")
            self.labels_added = True
        else:
            self.viewer.layers["Selected ROI"].data = np.array(self.labels).astype('int8')
 
    def onHeaderClicked(self, logicalIndex):
        
        if (self.batched):
            show_info('Feature not enabled for batched processing')
            return

        column = self.result.columns[logicalIndex]
        
        self.slider_feature_name = column
        
        max = self.result[column].max()
        min = self.result[column].min()
        
        self.slider_layer_name = column + " in " + self.segmentation.name
        
        labels = self.result.iloc[:,2]
        values = self.result.iloc[:,logicalIndex]
        self.label_values = pd.Series(values, index=labels).to_dict()
        
        self.colormap = None
        for layer in self.viewer.layers:
            if layer.name == self.slider_layer_name:
                self.colormap = layer.data 
                
        if (self.colormap is None):
            raise RuntimeError(f"Colormap layer {self.slider_layer_name} not found.")
                
        self._add_range_slider(min, max)
    
    
    def _get_label_from_range(self, min_bound, max_bound):

        for ix, iy in np.ndindex(self.colormap.shape):
            
            if (self.seg[ix, iy] != 0):
                
                if(np.isnan(self.label_values[int(self.seg[ix, iy])])):
                    continue
                
                value = self.label_values[int(self.seg[ix, iy])]
                
                if (value <= max_bound and value >= min_bound):
                    self.labels[ix, iy] = self.seg[ix, iy]
                    self.colormap[ix, iy] = value
                else:
                    self.labels[ix, iy] = 0
                    self.colormap[ix, iy] = 0
                
       
        self.viewer.layers[self.slider_layer_name].data = np.array(self.colormap)
            
             
        if (not self.labels_added):
            self.viewer.add_labels(np.array(self.labels).astype('int8'), name="Selected ROI")
            self.labels_added = True
            
        else:
            self.viewer.layers["Selected ROI"].data = np.array(self.labels).astype('int8')
            
    
    def _add_range_slider(self, min_value, max_value):

        min_value = util.round_down_to_5_sig_figs(min_value)
        max_value = util.round_up_to_5_sig_figs(max_value)
        
        if (self.slider_added):

            self.slider.setRange(min_value, max_value)
            self.range = [min_value, max_value]
            self.slider.setValue([min_value, max_value])
            self.name_label.setText(self.slider_feature_name)
            self.min_box.setText(str(min_value))
            self.max_box.setText(str(max_value))
            
        else:  

            self.slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
            self.range = [min_value, max_value]
            self.slider.setRange(min_value, max_value)
            self.slider.setValue([min_value, max_value])
            self.slider.valueChanged.connect(self._update_slider)

            layout = QVBoxLayout()
            self.name_label = QLineEdit(self.slider_feature_name)
            self.name_label.setAlignment(Qt.AlignCenter)
            self.name_label.setReadOnly(True)
            layout.addWidget(self.name_label)
            
            layout.addWidget(self.slider)
            
            widget = QWidget()
            widget.setLayout(layout)
            self.dock_widget = self.viewer.window.add_dock_widget(widget)
            
            self.min_box = QLineEdit(str(min_value))
            self.min_box.setReadOnly(False)
            self.min_box.textChanged.connect(self._get_minimum_text)

            self.max_box = QLineEdit(str(max_value))
            self.max_box.setReadOnly(False)
            self.max_box.textChanged.connect(self._get_maximum_text)

            
            self.hlayout = QHBoxLayout()
            self.hlayout.addWidget(self.min_box)
            self.hlayout.addWidget(self.max_box)
            layout.addLayout(self.hlayout)
            
            # Add a label to the dock widget to display text at the top
            self.label = QLabel("Adjust Range")
            self.label.setAlignment(Qt.AlignCenter)
            self.dock_widget.setTitleBarWidget(self.label)

            self.slider_added = True

        
    def _update_slider(self, event):
        
        min_value = util.round_down_to_5_sig_figs(event[0])
        max_value = util.round_up_to_5_sig_figs(event[1])
        
        self.min_box.setText(str(min_value))
        self.max_box.setText(str(max_value))
        
        self._get_label_from_range(min_value, max_value)

    def _get_minimum_text(self):
        user_input = self.min_box.text()

        try :
            value = float(user_input)
            if (value >= self.range[0] and value <= self.range[1]):
                self.slider.setValue([value, float(self.max_box.text())])
        except:
            return
        
    def _get_maximum_text(self):
        user_input = self.max_box.text()

        try: 
            value = float(user_input)
            if (value >= self.range[0] and value <= self.range[1]): 
                self.slider.setValue([float(self.min_box.text()), value])
        
        except:
            return
            
    