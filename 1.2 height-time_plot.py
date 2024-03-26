# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:38:18 2024

@author: WIN 11
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib
import os
import sunpy.map
import matplotlib.ticker as ticker

data_list = os.listdir('Data/AIA304/2012')
AIA_304 = f'Data/AIA304/2012/{data_list[160]}'
aia_map = sunpy.map.Map(AIA_304) #to ensure that sdoaia304 cmap has reloaded
sun_rad = aia_map.meta['RSUN_OBS'] #in arcsec
sun_rad_ref = aia_map.meta['RSUN_REF']/1000 #in km
one_arcsec_to_km = sun_rad_ref/sun_rad

intensity = np.load(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\intensity_along_line\intensity.npy")
dt_intensity = pd.read_csv(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\intensity_along_line\datetime_intensity.csv", index_col='Unnamed: 0',parse_dates=['date'], date_format='%Y-%m-%dT%H:%M:%S.%f')
dt_intensity['minute'] = dt_intensity['date'].apply(lambda i:i - dt_intensity['date'][0])
dt_intensity['minute'] = dt_intensity['minute'].apply(lambda i:i.seconds/60)
ang_sep = pd.read_csv(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\intensity_along_line\angular_separation.csv")
ang_sep['h_km'] = ang_sep.h_arcsec*one_arcsec_to_km
# df = pd.read_csv(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\intensity_along_line\obsolet\height_Su 2015_np minmax.csv", index_col='Unnamed: 0')
# df = df.reset_index(drop=True) #needed because I delete one row manually. 

bin_factor = int(np.round(intensity.shape[0] / 300))  # Calculate the bin factor
image_binned = np.mean(intensity.reshape(intensity.shape[0] // bin_factor, bin_factor, intensity.shape[1] // bin_factor, bin_factor), axis=(1, 3))
target_shape = (int(np.round(intensity.shape[0] / bin_factor / 2)) * 2, int(np.round(intensity.shape[1] / bin_factor / 2)) * 2)  # ~2" resolution

# Assuming height_time_image is your input image
# Step 1: Multiply each row by its height
image_binned[:, 1] *= image_binned[:, 0]
# Step 2: Calculate the median value
median_value = np.median(image_binned)
# Step 3: Threshold the image above 1.5x the median value
threshold_value = 0.01 * median_value
image_binned[image_binned < threshold_value] = np.nan
width = 6
height = width / 0.618
fig, ax = plt.subplots()
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)

orig_map=matplotlib.colormaps.get_cmap('sdoaia304')
reversed_map = orig_map.reversed()
# plt.imshow(intensity, cmap=reversed_map, origin ='lower',vmin=-5, vmax=35)
xlim = [dt_intensity.date[0],dt_intensity.date[len(dt_intensity)-1]]
x_lim = mdates.date2num(xlim)
ax.imshow(image_binned, cmap=reversed_map, origin='lower', vmax=35, extent=[x_lim[0],x_lim[1],0,ang_sep.h_arcsec.max()], aspect='auto')
# x_label_list = ['A1', 'B1', 'C1', 'D1']
ax.set_ylim(0,470)
ax.set_ylabel('Distance along slice (arcsec)')
ax.set_xlabel('Start time = {}'.format(dt_intensity.date[0].strftime('%Y/%m/%d %H:%M:%S')))

ax.text(dt_intensity.date[10],410,'(b)',fontsize=20)

ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
ax.xaxis_date()
date_format = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=80))
ax.axvline(dt_intensity.date[151],0,1, color = 'black', ls = '--')

ax2 = ax.secondary_xaxis('top')
ax2.xaxis.set_tick_params(labeltop=False)
ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=80))

ax3 = ax.secondary_yaxis('right')
ax3.yaxis.set_tick_params(labelright=False)
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(10))
# plt.axis('off') # for plain plot
# plt.savefig('Results/height-time_plot/fig_6b.png',bbox_inches='tight', dpi=100)
plt.show()

# plt.close()

