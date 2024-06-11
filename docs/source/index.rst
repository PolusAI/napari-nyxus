Nyxus Napari
------------

Nyxus Napari is a Napari plugin for running feature calculations on image-segmentation image pairs, using the
Nyxus application to compute features. Nyxus is a feature-rich, highly optimized, Python/C++ application capable 
of analyzing images of arbitrary size and assembling complex regions of interest (ROIs) split across multiple image tiles and files. 

For more information on Nyxus, see https://github.com/PolusAI/nyxus.
 
Installation
============

To install Napari, it is recommended to first create a separate Conda environment. 

.. code-block:: bash

   conda create -y -n napari-env -c conda-forge python=3.9
   conda activate napari-env


After creating the Conda environment,
install Napari using pip 

.. code-block:: bash

   python -m pip install "napari[all]"
   python -m pip install "napari[all]" --upgrade


or using conda

.. code-block:: bash

   conda install -c conda-forge napari
   conda update napari


Next, Nyxus must be installed. Note that the version of Nyxus must be greater than or equal to `0.50` to run the Napari plugin.

.. code-block:: bash

   pip install nyxus


or build from source using the instructions at https://github.com/PolusAI/nyxus#building-from-source using the conda build for the
python API.

After installing Napari and Nyxus, the Nyxus Napari plugin can be installed by cloning this repo and then building the plugin from the source. 
An example of this process is provided below.

.. code-block:: bash

   git clone https://github.com/PolusAI/napari-nyxus.git
   cd napari_nyxus
   pip install -e .


Napari can then be ran by running 

.. code-block:: bash

   napari


Use
===
After installing the plugin, start Napari by running the command `napari` from the command line. Once the Napari 
GUI is loaded, the Nyxus plugin can be loaded from the `Plugins` menu in the toolbar by going to Plugins -> nyxus-napari.

.. image:: https://github.com/PolusAI/napari-nyxus/raw/main/docs/source/img/plugin_menu.png

A widget will appear in the Napari viewer to run Nyxus.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   Examples

