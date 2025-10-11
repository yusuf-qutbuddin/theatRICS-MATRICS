# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:19:29 2023

@author: Krohn
"""

import pandas as pd
import numpy as np
import os

path = r'\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\Experiment_analysis\20231221_JHK_SC_FCS_GUVs_Coatings\LS-FCS_fitting\g2diffOffsetLS-FCSACF_blcorr.csv'


# Load and unpack Kristine format FCS data
data = pd.read_csv(path, header = 0)


# Sort the important information
file_identifiers = []
px_numbers = []
channel_numbers = []

n_fits = data.shape[0]

for i_fit in range(n_fits):
    
    _, tail = os.path.split(data.Filename[i_fit])
    try:     
        # works for three-digit pixel numbers
        px_numbers.append(int(tail[-14:-11]))
        file_identifiers.append(tail[:-21])
        channel_numbers.append(int(tail[-18]))
    except:
            
        try:
            # works for two-digit pixel numbers
            px_numbers.append(int(tail[-13:-11]))
            file_identifiers.append(tail[:-20])
            channel_numbers.append(int(tail[-17]))
        except:
            # If the logic for two-digit nubers failed, we have a single-digit number
            px_numbers.append(int(tail[-12]))
            file_identifiers.append(tail[:-19])
            channel_numbers.append(int(tail[-16]))
            
max_px = np.max(px_numbers)

data['File_ID'] = file_identifiers
data['Px_Number'] = px_numbers
data['Channel'] = channel_numbers

is_used = np.zeros(n_fits, dtype = np.bool8)

for i_fit in range(n_fits):
    
    # Did we use this one already?
    if not is_used[i_fit]:
        
        # Not used yet. Then let's assemble the data from this fit, and the others that belong to it.
        this_file_id = data['File_ID'][i_fit]
        this_channel_no = data['Channel'][i_fit]
        this_px_no = data['Px_Number'][i_fit]

        tau_d = np.ones(max_px, dtype = np.float64)
        tau_d[:] = np.nan
        n_particles = np.ones(max_px, dtype = np.float64)
        n_particles[:] = np.nan
        acr = np.ones(max_px, dtype = np.float64)
        acr[:] = np.nan
        cpp_avg = np.ones(max_px, dtype = np.float64)
        cpp_avg[:] = np.nan
        
        # Sanity-check if this fit was any good
        if ((data['Tau diffusion'][i_fit] > 1E-5) and
            (data['Tau diffusion'][i_fit] < 0.2) and
            (data['N'][i_fit] > 0)):
            
            tau_d[this_px_no-1] = data['Tau diffusion'][i_fit]
            n_particles[this_px_no-1] = data['N'][i_fit]
            cpp_avg[this_px_no-1] = data['CPP average'][i_fit]
        
        # ACR is different, we write that even for a bad fit
        acr[this_px_no-1] = data['Count Rate'][i_fit]

        # Leave a marker that we are done with this one
        is_used[i_fit] = True

        # Searching for the other pixels
        for i_other in range(i_fit+1, n_fits):
            if ((not is_used[i_other]) and 
            (data['File_ID'][i_other] == this_file_id) and 
            (data['Channel'][i_other] == this_channel_no)):
                
                other_px_no = data['Px_Number'][i_other]
                # Sanity-check if this fit was any good
                if ((data['Tau diffusion'][i_other] > 1E-5) and
                    (data['Tau diffusion'][i_other] < 0.2) and
                    (data['N'][i_other] > 0)):

                    tau_d[other_px_no-1] = data['Tau diffusion'][i_other]
                    n_particles[other_px_no-1] = data['N'][i_other]
                    cpp_avg[other_px_no-1] = data['CPP average'][i_other]
                    
                # ACR is different, we write that even for a bad fit
                acr[other_px_no-1] = data['Count Rate'][i_other]
                
                # Leave a marker that we are done with this one
                is_used[i_other] = True

        # We should have all now. Re-assemble into its own dataframe and save
        out_df = pd.DataFrame({'Diffusion_times': tau_d,
                                   'N': n_particles,
                                   'ACR': acr,
                                   'CPP_avg': cpp_avg})
        
        out_df.to_csv(os.path.join(os.path.split(path)[0], 
                                       this_file_id + '_Ch' + str(this_channel_no) + '_sorted_fit_params.csv'),
                                   header = True, 
                                   index = True)
        
        