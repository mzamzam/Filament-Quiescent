# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:09:42 2024

@author: MZN
modified from https://www.blackbox.ai/share/e0ceae40-9f5b-4444-b7ea-31d564db4640
"""

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import sunpy.map
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# df = pd.read_csv(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\intensity_along_line\height_Su 2015_np minmax.csv", index_col='Unnamed: 0')
# df = df.reset_index(drop=True) #needed because I delete one row manually. 
data_list = os.listdir('Data/AIA304/2012')
AIA_304 = f'Data/AIA304/2012/{data_list[0]}'
aia_map = sunpy.map.Map(AIA_304)
sun_rad = aia_map.meta['RSUN_OBS'] #in arcsec
sun_rad_ref = aia_map.meta['RSUN_REF']/1000 #in km
one_arcsec_to_km = sun_rad_ref/sun_rad
scale = aia_map.scale[0].value

ang_sep = pd.read_csv(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\intensity_along_line\angular_separation.csv")
ang_sep['h_km'] = ang_sep.h_arcsec*one_arcsec_to_km

dt_intensity = pd.read_csv(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\intensity_along_line\datetime_intensity.csv", index_col='Unnamed: 0',parse_dates=['date'], date_format='%Y-%m-%dT%H:%M:%S.%f')
dt_intensity['minute'] = dt_intensity['date'].apply(lambda i:i - dt_intensity['date'][0])
dt_intensity['minute'] = dt_intensity['minute'].apply(lambda i:i.seconds/60)

edge_coord_all = pd.read_csv(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\canny\edge_coord.csv", index_col='Unnamed: 0')
edge_coord = pd.read_csv(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\canny\edge_coord.csv", index_col='Unnamed: 0')
edge_coord_all['x_minute'] = edge_coord_all['x'].apply(lambda i: dt_intensity.loc[i].minute)
edge_coord_all['y_Mm'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_km)/1000
edge_coord['x_minute'] = edge_coord['x'].apply(lambda i: dt_intensity.loc[i].minute)
edge_coord['x_second'] = edge_coord.x_minute*60
edge_coord['y_Mm'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_km)/1000
edge_coord['y_arcsec'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_arcsec)
edge_coord.drop_duplicates(subset=['x'], inplace=True)

t_data = edge_coord.x_minute[:73]
h_data = edge_coord.y_Mm[:73] #different data slicing resulting in different result

def model_h(t,c0,tau,c1,c2,t0):
    return c0 * np.exp((t-t0)/ tau) + c1 * (t-t0) + c2
def t_onset(tau,c1,c0,t0):
    return tau*np.log(c1*tau/c0)+t0
# popt, pcov = curve_fit(model_h, t_data, h_data, p0=[1, 30, 1, 60, 30])
popt, pcov = curve_fit(model_h, t_data, h_data, p0=[7,31,0.1,89,211]) #p0=[355,128,0,336]
c0_opt, tau_opt,c1_opt,c2_opt,t0_opt = popt
t_model = np.linspace(min(t_data), max(t_data), 100)
h_model = model_h(t_model,c0_opt,tau_opt,c1_opt,c2_opt, t0_opt)
print(f'selisih min: {h_model.min()-h_data.min()}')
print(f'selisih max: {h_model.max()-h_data.max()}')
print(f'var c0: {c0_opt}')
print(f'var tau: {tau_opt}')
print(f'var c1: {c1_opt}')
print(f'var c2: {c2_opt}')
print(f'var t0: {t0_opt}')

t_onset = t_onset(tau_opt, c1_opt, c0_opt, t0_opt)
start_time = dt_intensity.loc[edge_coord.x.min()].date
onset_time = start_time + timedelta(minutes=t_onset)
print(f'onset time: {onset_time}')
if t_onset in edge_coord.x_minute: 
    onset_height = edge_coord[edge_coord.x_minute == int(t_onset)].y_Mm.values[0]
    print(f'onset height: {onset_height} Mm')
else:
    print('===================================')
    print(f'Cannot find the onset height of onset time = {t_onset} minute.')
    print('Find the closest onset time on the coordinate list (edge_coord).')
    print('Then, run again the onset_height syntax (line 78).')
    print('==========================')
    onset_height = edge_coord[edge_coord.x_minute == 186].y_Mm.values[0]

fig, ax = plt.subplots()
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)

t = t_data.apply(lambda i:start_time + timedelta(minutes=i))
dft_model = pd.Series(t_model)
model_t = dft_model.apply(lambda i:start_time + timedelta(minutes=i))

### height (Mm) plot
# ax.plot(t, h_data)
ax.plot(model_t,h_model, color='k')
ax.set_ylim(0)
ax.set_ylabel('Height (Mm)')
ax.set_xlabel('Start time = {}'.format(dt_intensity.date[0].strftime('%Y/%m/%d %H:%M:%S')))

ax.text(dt_intensity.date[5],285,'(d)',fontsize=20)

ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

# ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
ax.xaxis_date()
date_format = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=80))
ax.axvline(onset_time, color = 'black', ls = '-.')
ax.axvline(dt_intensity.date[148], color = 'black', ls = '--')

ax2 = ax.secondary_xaxis('top')
ax2.xaxis.set_tick_params(labeltop=False)
# ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=80))

ax3 = ax.secondary_yaxis('right')
ax3.yaxis.set_tick_params(labelright=False)
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(10))
# plt.axis('off') # for plain plot
# plt.savefig('Results/fitting_height/fig_6d.png',bbox_inches='tight', dpi=100)
plt.show()

# plt.close()



