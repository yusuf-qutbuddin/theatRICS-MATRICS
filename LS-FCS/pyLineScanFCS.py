# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:23:37 2024

@author: Krohn
"""

import os
from pylibCZIrw import czi as pyczi
import numpy as np
import multipletau
import matplotlib.pyplot as plt

# File
folder = r'D:\temp'
file = r'GUV_3-1_lsFCS.czi'


'''
Raw data contains ca. 200k line scans of the same position at the highest speed allowed by the LSM980 galvo.

The idea is to auto-correlate the signal over time of each pixel over time, 
so in this case the line-scan FCS is more of a parallelized FCS than spatiotemporal scanning FCS.

Strangely, I can open the data just fine in ZEN, but when I attempt to open the 
same data in ImageJ or through this script, most time points are empty (-> see variable
"nonzero_frames": For me at least, contains only 2333 elements rather than the expected 199961)

Conda environment used: see environment.yml
ZEN version: 3.9.101.04000
ImageJ version: 1.54f
    
    

'''

path = os.path.join(folder, file)
with pyczi.open_czi(path) as czidoc:
    
    # Here's a list of attributes and methods to 
    # try out when one wants to change stuff
    CZIreader_methods = [method_name for method_name in dir(czidoc)]
    
    # Read out some metadata
    metadata = czidoc.metadata['ImageDocument']['Metadata'] # These two layers are pro-forma, there is nothing else in here
    line_time_s = float(metadata['Information']['Image']['Dimensions']['T']['Positions']['Interval']['Increment'])
    
    print(f'Data type: {czidoc.get_channel_pixel_type(0)}')
    
    # get the image dimensions as a dictionary, where the key identifies the dimension
    total_bounding_box = czidoc.total_bounding_box
    total_bounding_rectangle = czidoc.total_bounding_rectangle
    
    '''
    total_bounding_rectangle == pyczi.Rectangle(total_bounding_box['X'][0],
                                                total_bounding_box['Y'][0],
                                                total_bounding_box['X'][1] - total_bounding_box['X'][0],
                                                total_bounding_box['Y'][1] - total_bounding_box['Y'][0])
    '''

    n_frames = total_bounding_box['T'][1]
    x_start = total_bounding_box['X'][0]
    x_stop = total_bounding_box['X'][1]
    n_x = x_stop - x_start
    print('Dimensions:')
    [print(f'{key}: {total_bounding_box[key]}') for key in total_bounding_box.keys()]


    traces = np.zeros((n_frames, n_x), 
                      dtype = np.uint16)
    
    nonzero_frames = []
    for i_frame in range(n_frames):
        data_frame = czidoc.read(roi = total_bounding_rectangle,
                                 plane = {'T':i_frame,
                                          'C':0})
        traces[i_frame, :] = data_frame.reshape((1, n_x))
        if data_frame.sum() > 0:
            nonzero_frames.append(i_frame)

        if i_frame % 10000 == 0:
            print(f'Reading: Frame {i_frame} of {n_frames}')

        
        
    # Now we close the file
    
    
    
    
traces = np.float_(traces)
sum_signal = traces.sum()
        
print(f'Sum of total photon counts is {sum_signal}.')



# # Get all correlation functions and overlay them

for i_px in range(n_x):
    res = multipletau.autocorrelate(traces[:,i_px], 
                                    m = 12, 
                                    normalize = True,
                                    dtype=np.float_)
    tau = res[:,0] * line_time_s
    acf = res[:,1]
    
    tau_mask = np.logical_and(tau > 1E-6,
                              tau < 3.)
    
    tau = tau[tau_mask]
    acf = acf[tau_mask]

    if i_px == 0:
        acfs = np.zeros((acf.shape[0], n_x))
        acfs_plot = np.zeros((acf.shape[0], n_x))
    
    acfs[:, i_px] = acf
    acfs_plot[:, i_px] = acf / np.percentile(acf, 95)
    


axim = plt.imshow(acfs_plot,
                  vmin = 0,
                  vmax = 1.)

        
        
