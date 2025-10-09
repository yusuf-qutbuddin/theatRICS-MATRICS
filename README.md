Modular RICS Analysis GUI
A modular, extensible graphical user interface for performing and analyzing Raster Image Correlation Spectroscopy (RICS) experiments. This toolkit supports simulation, data import/export, and advanced analysis with a user-friendly workflow designed for membrane biophysics, imaging, and fluorescence correlation studies.

Features
Flexible RICS simulations: Isotropic, anisotropic, and rotated diffusion models using highly optimized Python backends.

Batch analysis: Fit and analyze real or simulated image stacks.

Progress monitoring: Responsive GUI with real-time progress and status bars.

Modular design: Easily extend with new simulation, import, or analysis modules.

Visualization: Integrated with Matplotlib for RICS map display and fitting results.

Quick Start
Prerequisites
Python 3.8+

Tkinter (comes with most Python installations)

NumPy, matplotlib, tifffile, Pillow

Install Python dependencies as follows:

text
pip install numpy matplotlib tifffile pillow
Installation
Clone the repository:

text
git clone https://github.com/<your-username>/modular-rics-gui.git
cd modular-rics-gui
(Optional) Set up a virtual environment for Python.

Running
From within the repository directory:

text
python modular_rics_gui.py
This launches the RICS Analysis GUI.

Usage
Simulation: Select the type and parameters in the GUI, then click 'Run Simulation'. You will see progress and status bars at the bottom of the window.

Analysis: Import real or simulated image stacks, fit RICS models, and export results.

Logging: The lower panel reports messages, warnings, and results.

For full workflow details and screenshots, see the Wiki.

File Overview
modular_rics_gui.py: Main GUI application script.

00_simRICS.py: Core module for RICS simulations.

01_Export_RICS_from_image.py, 02_RICS_fit.py: Additional modules for exporting RICS maps and fitting.

(See code comments for details on modular imports/architecture.)

Technical Notes
The GUI relies on multiprocessing.Queue for parallel simulation and thread-safe progress updates.

All UI updates from background threads are routed through the Tkinter after() mechanism to maintain responsiveness and avoid crashes.

Full source code is documented for easy extension and integration into custom pipelines.

Contributing
Contributions, feature requests, and bug reports are welcome! Please file issues and pull requests via GitHub.

License
MIT License (see LICENSE file for details)

Citation
If you use this tool in your research, please cite as:
