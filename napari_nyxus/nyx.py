import napari
from napari.layers import Image
from magicgui import magic_factory

from napari_nyxus.nyx_napari import Features


logo = "docs/source/img/polus.png"

@magic_factory(
    label_head=dict(
        #widget_type="Label", label=f'<h1><img src="{logo}" height="50" width="50"> Nyxus</h1>'
        widget_type="Label", label=f'<h1>Nyxus</h1>'
    ),
    documentation=dict(
        widget_type="Label", label=f"For information on the provided features see the <a href='https://nyxus.readthedocs.io/en/latest/featurelist.html'>documentation</a>"
    )
)
def widget_factory(
    label_head,
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
    GPU_id: int = 0,
    documentation = ""
    ):
    
    #wait for function call to load large modules
    from napari_nyxus import nyx_napari
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
    
    nyx.run()

    
    
    
    