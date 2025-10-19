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

 Follow the simple steps to use the GUI. 

 1. Activate the environment: `conda activate <ENVNAME>`
 2. Go to the correct installation directory and run `python theatRICS_gui.py`

 This should launch the GUI with the following tabs loaded. 

 1. Simulation tab

 The simulation tab contains the necessary parameters to use for simulating a raster scanned image for particles diffusing in 2D. 
 
 ![Simulation tab](/screenshots/Simulation_tab.png)

 The simulation can be run for isotropic and anisotropic diffusion. For isotropic case, choose the same diffusion coefficient for x and y. 

 
 2. RICS export tab

 The window size is necessary for moving average correction, and **must** be an odd number. 
 
 ![Export tab](/screenshots/Export_tab.png)

 
 
 3. RICS fitting tab

 The fitting tab requires information of the PSF and the imaging parameters (these parameters can also be provided by uploading a czi file directly). It also has a feature for getting diffusion maps where window size is 
 the smaller ROIs over which RICS export and fitting will take place, and the offset is the overlap between individual ROIs. Here the offset should always be less than the window size, preferably offset = 0.5*window size.
 
 ![Fitting tab](/screenshots/Fitting_tab.png)


 4. Results and log tab

 This is where the results and logging takes place. 

 
