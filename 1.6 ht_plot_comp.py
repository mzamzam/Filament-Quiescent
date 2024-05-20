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
from skimage import feature
from datetime import timedelta
from scipy.optimize import curve_fit

# dtdt = '20230420'
dtdt = '20120311'
ins = 'AIA304'
data_list = os.listdir(f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}')
AIA_304 = f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}/{data_list[200]}'
# data_list = os.listdir('Data/AIA304/2012')
# AIA_304 = f'Data/AIA304/2012/{data_list[160]}'
aia_map = sunpy.map.Map(AIA_304) #to ensure that sdoaia304 cmap has reloaded
sun_rad = aia_map.meta['RSUN_OBS'] #in arcsec
sun_rad_ref = aia_map.meta['RSUN_REF']/1000 #in km
one_arcsec_to_km = sun_rad_ref/sun_rad

intensity = np.load(f"Results/intensity_along_line/intensity_{dtdt}.npy")
dt_intensity = pd.read_csv(f"Results/intensity_along_line/datetime_intensity_{dtdt}.csv", parse_dates=['date'], date_format='%Y-%m-%dT%H:%M:%S.%f')
dt_intensity['minute'] = dt_intensity['date'].apply(lambda i:i - dt_intensity['date'][0])
dt_intensity['minute'] = dt_intensity['minute'].apply(lambda i:i.seconds/60)

ang_sep_mid = pd.read_csv(f"Results/intensity_along_line/ang_sep_{dtdt}.csv")
ang_sep_be = pd.read_csv(f"Results/intensity_along_line/ang_sep_be_{dtdt}.csv")
ang_sep_ab = pd.read_csv(f"Results/intensity_along_line/ang_sep_ab_{dtdt}.csv")

ang_sep = ang_sep_mid
ang_sep['h_km'] = ang_sep.h_arcsec*one_arcsec_to_km

### syntax dari 1.3
arr = feature.canny(intensity, sigma=11)
coord = np.argwhere(arr==1)
coord_edge = coord[coord[:,0] > 50]
coordy, index = np.unique(coord_edge[:,0], return_index=True)
coord_edge = coord_edge[index]
df_coord_edge = pd.DataFrame(coord_edge, columns=['y', 'x'])
df_coord_edge['y_arcsec'] = df_coord_edge['y'].apply(lambda i:ang_sep.loc[i].h_arcsec)
df_coord_edge['x_minute'] = df_coord_edge['x'].apply(lambda i: dt_intensity.loc[i].minute)
df_coord_edge['y_Mm'] = df_coord_edge['y'].apply(lambda i:ang_sep.loc[i].h_km)/1000
df_coord_edge.drop_duplicates(subset=['x'], inplace=True)
start_time = dt_intensity.loc[df_coord_edge.x.min()].date
coord_edge_time = df_coord_edge.x_minute.apply(lambda i:start_time + timedelta(minutes=i))
####
### syntax from 1.4
def model_h(t,c0,tau,c1,c2,t0):
    return c0 * np.exp((t-t0)/ tau) + c1 * (t-t0) + c2
def t_ons(tau,c1,c0,t0):
    return tau*np.log(c1*tau/c0)+t0
t_data = df_coord_edge.x_minute[:73]
h_data = df_coord_edge.y_arcsec[:73]
popt, pcov = curve_fit(model_h, t_data, h_data, p0=[7,31,0.1,89,211]) #p0=[355,128,0,336]
c0_opt, tau_opt,c1_opt,c2_opt,t0_opt = popt
t_model = np.linspace(min(t_data), max(t_data), 100)
h_model = model_h(t_model,c0_opt,tau_opt,c1_opt,c2_opt, t0_opt)
dft_model = pd.Series(t_model)
model_t = dft_model.apply(lambda i:start_time + timedelta(minutes=i))
t = t_data.apply(lambda i:start_time + timedelta(minutes=i))

t_onset = t_ons(tau_opt, c1_opt, c0_opt, t0_opt)
onset_t = t_model[t_model >= t_onset][0]
start_time = dt_intensity[dt_intensity.minute >= t_model[0]].date.min()
onset_time = start_time + timedelta(minutes=onset_t)
onset_height = df_coord_edge[df_coord_edge.x_minute >= onset_t].y_Mm.values[0]
onset_height_arcsec = df_coord_edge[df_coord_edge.x_minute >= onset_t].y_arcsec.values[0]
onset_h = h_model[t_model >= onset_t][0]
onset_h_Mm = df_coord_edge[df_coord_edge.y_arcsec>= onset_h].y_Mm.values[0]

print('onset time: {} UT'.format(onset_time.strftime('%Y/%m/%d %H:%M')))
print(f'onset height: {onset_h_Mm} Mm')
print(f'onset height: {onset_h} arcsec')
###

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
fig, ax = plt.subplots()
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)

orig_map=matplotlib.colormaps.get_cmap('sdoaia304')
reversed_map = orig_map.reversed()
xlim = [dt_intensity.date[0],dt_intensity.date[len(dt_intensity)-1]]
# end_time = start_time + timedelta(minutes=t_data.values[-1])
# xlim = [start_time,end_time]
x_lim = mdates.date2num(xlim)
ax.imshow(image_binned, cmap=reversed_map, origin='lower', vmax=35, extent=[x_lim[0],x_lim[1],0,ang_sep.h_arcsec.max()], aspect='auto')
# ax.plot(coord_edge_time[3:], df_coord_edge.y_arcsec[3:], color='k')
ax.plot(model_t[10:], h_model[10:], color='k')
ax.set_ylim(0,470)
ax.set_ylabel('Distance along slice (arcsec)')
ax.set_xlabel('Start time = {}'.format(start_time.strftime('%Y/%m/%d %H:%M:%S')))

ax.text(dt_intensity.date[10],410,'(b)',fontsize=20)
ax.text(onset_time-timedelta(minutes=45),onset_h+13,f'{round(onset_h_Mm)} Mm',fontsize=13)
ax.plot(onset_time,onset_h, 'k*', ms=10)

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
# plt.savefig('Results/ht_plot_comp/fig_6b.png',bbox_inches='tight', dpi=100)
plt.show()

# plt.close()

