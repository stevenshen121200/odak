"""
``odak``
===================
Odak is the fundamental Python library needed for scientific computing in optical sciences. It includes modules for geometric raytracing and wave optics.

"""
try:
    import cupy as np
except:
    import numpy as np
import odak.tools
import odak.raytracing
import odak.wave
import odak.visualize
import odak.manager
import odak.jones

version_info = 0, 13, 0
__version__  = '.'.join(map(str, version_info))
