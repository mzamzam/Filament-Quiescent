# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:34:54 2024

@author: WIN 11
"""

import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
from skimage import feature
import os
import sunpy.map
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import timedelta

data_list = os.listdir('Data/AIA304/2012')
AIA_304 = f'Data/AIA304/2012/{data_list[160]}'
aia_map = sunpy.map.Map(AIA_304)
sun_rad = aia_map.meta['RSUN_OBS'] #in arcsec
sun_rad_ref = aia_map.meta['RSUN_REF']/1000 #in km
one_arcsec_to_km = sun_rad_ref/sun_rad

intensity = np.load(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\intensity_along_line\intensity.npy")
png = r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\height-time_plot\fig_6b_polos.png"

dt_intensity = pd.read_csv(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\intensity_along_line\datetime_intensity.csv", index_col='Unnamed: 0',parse_dates=['date'], date_format='%Y-%m-%dT%H:%M:%S.%f')
dt_intensity['minute'] = dt_intensity['date'].apply(lambda i:i - dt_intensity['date'][0])
dt_intensity['minute'] = dt_intensity['minute'].apply(lambda i:i.seconds/60)
ang_sep = pd.read_csv(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\intensity_along_line\angular_separation.csv")
ang_sep['h_km'] = ang_sep.h_arcsec*one_arcsec_to_km

orig_map = plt.colormaps.get('sdoaia304')
reversed_map = orig_map.reversed()

arr = feature.canny(intensity, sigma=11)

fig, ax = plt.subplots()
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)

## Detect contour all edges from array numpy with shape 182x307
coord = np.argwhere(arr==1)
df_coord = pd.DataFrame(coord, columns=['y', 'x'])
df_coord['y_arcsec'] = df_coord['y'].apply(lambda i:ang_sep.loc[i].h_arcsec)
df_coord['x_minute'] = df_coord['x'].apply(lambda i: dt_intensity.loc[i].minute)

## Detect certain edges from array numpy with shape 182x307
coord_edge = coord[coord[:,0] > 50]
# coord_edge = coord[coord[:,1] < 170]
coordy, index = np.unique(coord_edge[:,0], return_index=True)
coord_edge = coord_edge[index]
df_coord_edge = pd.DataFrame(coord_edge, columns=['y', 'x'])
df_coord_edge['y_arcsec'] = df_coord_edge['y'].apply(lambda i:ang_sep.loc[i].h_arcsec)
df_coord_edge['x_minute'] = df_coord_edge['x'].apply(lambda i: dt_intensity.loc[i].minute)
# df_coord_edge.to_csv('Results/canny_edge/edge_coord.csv')
# xlim = [dt_intensity.date[0],dt_intensity.date[len(dt_intensity)-1]]
# x_lim = mdates.date2num(xlim)
# , extent=[x_lim[0],x_lim[1],0,ang_sep.h_arcsec.max()]
# ax.imshow(arr,origin='lower', cmap='binary', aspect='auto')

# ax.plot(coord[:, 1], coord[:, 0], color='cyan', marker='.', 
#         linestyle='None', markersize=2)
# ax.plot(coord_edge[:, 1], coord_edge[:, 0], color='red', marker='.',
#          linestyle='None', markersize=2)
start_time = dt_intensity.loc[df_coord_edge.x.min()].date
coord_time = df_coord.x_minute.apply(lambda i:start_time + timedelta(minutes=i))
coord_edge_time = df_coord_edge.x_minute.apply(lambda i:start_time + timedelta(minutes=i))
ax.plot(coord_time, df_coord.y_arcsec, color='cyan', marker='.',
          linestyle='None', markersize=2)
ax.plot(coord_edge_time, df_coord_edge.y_arcsec, color='red', marker='.',
          linestyle='None', markersize=2)
ax.set_ylim(0,450)
ax.set_ylabel('Height (arcsec)')
ax.set_xlabel('Start time = {}'.format(dt_intensity.date[0].strftime('%Y/%m/%d %H:%M:%S')))

ax.text(dt_intensity.date[7],400,'(c)',fontsize=20)

ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
ax.xaxis_date()
date_format = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=80))
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(20))
ax.axvline(dt_intensity.date[151],0,1, color = 'black', ls = '--')

ax2 = ax.secondary_xaxis('top')
ax2.xaxis.set_tick_params(labeltop=False)
ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=80))

ax3 = ax.secondary_yaxis('right')
ax3.yaxis.set_tick_params(labelright=False)
ax3.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(20))
plt.tight_layout()
plt.savefig('Results/canny/fig_6c.png', dpi=200)
plt.show()



