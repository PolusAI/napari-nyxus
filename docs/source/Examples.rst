Examples
--------

Running Feature Calculations
============================

After loading the Napari GUI, the Nyxus plugin can be loaded from the `Plugins` menu in the toolbar by going to Plugins -> nyxus-napari.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/plugin_menu.png

A widget will appear in the Napari viewer to run Nyxus.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/nyxus_loaded.png

As shown by the example above, Nyxus will take in Intensity and Segmentation images. These parameters can either be a stack
of images or a single image pair. To load an image pair, use File -> Open File(s)... and select the images to load.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/open_image.png 

Note that this method can also be used to open a stack of image, by using File -> Open Folder... instead of images. 

If the segmentation is loaded as an Image type in the napari viewer, it must first be converted to the Labels type. The image can converted as shown below.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/convert_to_labels.png

The loaded files can then be selected with the Intensity and Segmentation drop down menus. Other parameters can also be changed,
such as which features to calculate. For more information on the available features, see https://nyxus.readthedocs.io/en/latest/featurelist.html.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/setup_calculation.png

After running Nyxus, the feature calculations will also appear in the Napari viewer.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/feature_results.png


Interacting with feature calculations
=====================================

The Nyxus Napari plugin provides functionality to interact with the table containing the feature calculations. First, click on the segmentation image and then select `show selected` in the layer controls. 


Then, if a value is clicked in the `label` column of the table, the respective ROI will be highlighted in the segmentation image in the viewer.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/click_label.png

To select the ROI and have it added to a separate Labels image, the label in the table can be double clicked. Each double clicked label will be added to the same Labels image as show below. To unselect, the ROI, double click its respective label again.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/double_click_label.png

This feature can also be used in the opposite way, i.e. if an ROI is clicked in the segmentation image, the respective row in the 
feature table will be highlighted.

If one of the column headers are double clicked, a colormap will be generated in the Napari viewer showing the values of the features in the clicked
column. For example, if `Intensity` features are calculated, the `INTEGRATED_INTENSITY` column can be clicked and the colormap will appear.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/feature_colormap.png

Once the colormap is loaded, a slider will appear in the window with the minimum value being the minimum value of the feature colormap and the 
maximum value of the slider is the maximum value of the colormap. By adjusting the ranges in the slider, a new label image will appear in the viewer
that contains the ROIs who's features values fall within the slider values.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/slider_feature.png

The new labels resulting from the range slider selector can then be used to run Nyxus on by using the labels image as the `Segmentation` parameter.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/run_on_colormap_labels.png


Feature calculation heat map
============================

To visualize how features change between images, ROIs, or annotations, a heat map can be added to the feature calculation table.
To add a heatmap, first run Nyxus on image-segmentation pair(s). Next, the heat map can be added through by first selecting a colormap option:

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/heatmap_drop_down.png

Next, it can be selected whether to toggle the option to remove the feature calculation values. Hiding the feature calculations values will hide the 
values in each cell and make each cell the same size. This can make visualization of the feature calculations easier.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/heatmap_toggle.png

After clicking the ``Generate Heatmap`` button, the feature calculation table will have the colormap applied.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/heatmap.png


Extracting annotations from filenames
=====================================

The Nyxus plugin also provides the ability to extract annotations from the image names. To extract annotations, the column name must first be selected.
After selecting the column name, a ``filepattern`` must be supplied to give the pattern of the annotation. For more information on what a filepattern is,
see https://github.com/PolusAI/filepattern. Finally, supply one of the variables to extract. 

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/annotation_input.png

After clicking ``Extract annotation``, the annotation will appear as a new column in the feature calculation table.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/annotation_column.png

Sorting the feature calculation table
=====================================

To sort the feature calculation table by a column, enter a column into the text box shown below. Note that this column can be any of the present columns,
including annotations. 

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/sort_box.png

After clicking ``Sort``, the table will be sorted by the selected column. To sort by multiple columns, enter the columns into the box in order, separated by
spaces.
