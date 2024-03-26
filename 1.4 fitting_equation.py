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

edge_coord_all = pd.read_csv(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\canny_edge\edge_coord.csv", index_col='Unnamed: 0')
edge_coord = pd.read_csv(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\canny_edge\edge_coord.csv", index_col='Unnamed: 0')
edge_coord_all['x_minute'] = edge_coord_all['x'].apply(lambda i: dt_intensity.loc[i].minute)
edge_coord_all['y_Mm'] = edge_coord['y']*one_arcsec_to_km/1000 # in Mm
edge_coord['x_minute'] = edge_coord['x'].apply(lambda i: dt_intensity.loc[i].minute)
edge_coord['x_second'] = edge_coord.x_minute*60
edge_coord['y_Mm'] = edge_coord['y']*scale*one_arcsec_to_km/1000 # in Mm
edge_coord['y_arcsec'] = edge_coord['y']*scale
edge_coord.drop_duplicates(subset=['x'], inplace=True)

t_data = edge_coord.x_minute[:73]
h_data = edge_coord.y_Mm[:73] #different data slicing resulting in different result

# def model_h(t,c0,tau,c1,c2,t0):
#     return c0 * np.exp((t-t0)/ tau) + c1 * (t-t0) + c2
def model_h(t,c0,tau,c1,c2,t0):
    return c0 * np.exp((t-t0)/ tau) + c1 * (t-t0) + c2
def t_onset(tau,c1,c0,t0):
    return tau*np.log(c1*tau/c0)+t0
# popt, pcov = curve_fit(model_h, t_data, h_data, p0=[1, 30, 1, 60, 30])
popt, pcov = curve_fit(model_h, t_data, h_data, p0=[1,31,0.283,115,138]) #p0=[355,128,0,336]
# c0_opt, tau_opt, c1_opt, c2_opt, t0_opt = popt
c0_opt, tau_opt,c1_opt,c2_opt,t0_opt = popt
t_model = np.linspace(min(t_data), max(t_data), 100)
# h_model = model_h(t_model, c0_opt, tau_opt, c1_opt, c2_opt, t0_opt)
h_model = model_h(t_model,c0_opt,tau_opt,c1_opt,c2_opt, t0_opt)
print(f'selisih min: {h_model.min()-h_data.min()}')
print(f'selisih max: {h_model.max()-h_data.max()}')
print(f'var c0: {c0_opt}')
print(f'var tau: {tau_opt}')
print(f'var c1: {c1_opt}')
print(f'var c2: {c2_opt}')
print(f'var t0: {t0_opt}')
plt.scatter(t_data, h_data)
plt.ylim(0)
plt.plot(t_model,h_model, color='r')
plt.show()

t_onset = t_onset(tau_opt, c1_opt, c0_opt, t0_opt)
# print(t_onset)
start_time = dt_intensity.loc[edge_coord.x.min()].date
onset_time = start_time + timedelta(minutes=t_onset)
onset_height = edge_coord[edge_coord.x_minute == np.round(t_onset)].y_Mm.values[0]
print(f'onset time: {onset_time}')
print(f'onset height: {onset_height} Mm')

## basic way of derivative
# dhdt = np.gradient(edge_coord.y_km[:73],edge_coord.x_second[:73])
# plt.plot(t_data, dhdt)
# dvdt = np.gradient(dhdt, edge_coord.x_second[:73])
# plt.plot(t_data, dvdt)

## utilize sympy
import sympy as smp
t,c0,tau,c1,c2,t0 = smp.symbols('t c0 tau c1 c2 t0', real=True)
h = c0 * smp.exp((t-t0)/tau) + c1*(t-t0) + c2
dhdt = smp.diff(h,t)
print(f'v(0) : {dhdt.subs([(t,8),(c0,c0_opt),(tau,tau_opt),(c1, c1_opt),(t0,t0_opt)]).evalf()}')
dhdt_f = smp.lambdify((t,c0,tau,c1,t0), dhdt)
hv = dhdt_f(t_model,c0=c0_opt,tau=tau_opt,c1=c1_opt,t0=t0_opt)
d2hdt2 = smp.diff(h,t,2)
d2hdt2_f = smp.lambdify((t,c0,tau,c1,t0), d2hdt2)
ha = d2hdt2_f(t_model,c0=c0_opt,tau=tau_opt,c1=c1_opt,t0=t0_opt)
# plt.plot(t_model, ha)
# plt.plot(t_model, hv)

