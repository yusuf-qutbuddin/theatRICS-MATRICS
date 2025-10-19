The software is being actively developed and currently has basic functionalities available, in case of any issues please contact yusufqq@biochem.mpg.de.

# Raster Image Correlation Spectroscopy simulation and analysis
A modular, extensible graphical user interface for performing and analyzing Raster Image Correlation Spectroscopy (RICS) experiments. This toolkit supports simulation, data import/export, and advanced analysis with a user-friendly workflow designed for membrane biophysics, imaging, and fluorescence correlation studies.
Currently the software is limited to Zeiss (.czi) files and TIFF files for the input image format for the raster scanned image. There will be a future update to involve other commonly used file types from other commercial microscope companies. 
## Features
**Flexible RICS simulations**: Isotropic, anisotropic, and rotated diffusion models.

**Analysis**: Fit and analyze real or simulated image stacks. (Non-GUI batch analysis)

**Progress monitoring**: Responsive GUI with real-time progress and status bars.

**Modular design**: Easily extend with new simulation, import, or analysis modules.

**Visualization**: Integrated with Matplotlib for RICS map display and fitting results.

## Installation

The current version has been tested to work through a conda environment with some necessary modules, you can also download anaconda [here](https://www.anaconda.com/). You can use the environment.yaml file for setting up the environment. 

**For Windows**: `conda env create --name <ENVNAME> --file environment_win.yml`

**For Linux**: `conda env create --name <ENVNAME> --file environment_lin.yml`

The software hasn't been tested for Mac but there is no particular reason why it shouldn't work. Once this test has been done, the environment file for Mac will also be uploaded. 

 ## Use

 The use of the python package

