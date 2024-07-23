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
dtdt = '20120312'
ins = 'AIA304'
data_list = os.listdir(f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}')
AIA_304 = f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}/{data_list[200]}'

# data_list = os.listdir('Data/AIA304/2012')
# AIA_304 = f'Data/AIA304/2012/{data_list[0]}'
aia_map = sunpy.map.Map(AIA_304)
sun_rad = aia_map.meta['RSUN_OBS'] #in arcsec
sun_rad_ref = aia_map.meta['RSUN_REF']/1000 #in km
one_arcsec_to_km = sun_rad_ref/sun_rad
scale = aia_map.scale[0].value

def model_h(t,c0,tau,c1,c2,t0):
    return c0 * np.exp((t-t0)/ tau) + c1 * (t-t0) + c2
def t_ons(tau,c1,c0,t0):
    return tau*np.log(c1*tau/c0)+t0

ang_sep_mid = pd.read_csv(f"Results/intensity_along_line/ang_sep_{dtdt}.csv")
ang_sep_be = pd.read_csv(f"Results/intensity_along_line/ang_sep_be_{dtdt}.csv")
ang_sep_ab = pd.read_csv(f"Results/intensity_along_line/ang_sep_ab_{dtdt}.csv")

dt_intensity = pd.read_csv(f"Results/intensity_along_line/datetime_intensity_{dtdt}.csv",parse_dates=['date'], date_format='%Y-%m-%dT%H:%M:%S.%f')
dt_intensity['minute'] = dt_intensity['date'].apply(lambda i:i - dt_intensity['date'][0])
dt_intensity['minute'] = dt_intensity['minute'].apply(lambda i:i.seconds/60)

edge_coord_mid = pd.read_csv(f"Results/canny/edge_coord_{dtdt}.csv", index_col='Unnamed: 0')
edge_coord_ab = pd.read_csv(f"Results/canny/edge_coord_ab_{dtdt}.csv", index_col='Unnamed: 0')
edge_coord_be = pd.read_csv(f"Results/canny/edge_coord_be_{dtdt}.csv", index_col='Unnamed: 0')

'''
## originl code before turn into function
#---mid---#
ang_sep = ang_sep_mid
ang_sep['h_km'] = ang_sep.h_arcsec*one_arcsec_to_km
edge_coord = edge_coord_mid

edge_coord['x_minute'] = edge_coord['x'].apply(lambda i: dt_intensity.loc[i].minute)
edge_coord['x_second'] = edge_coord.x_minute*60
edge_coord['y_Mm'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_km)/1000
edge_coord['y_arcsec'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_arcsec)
edge_coord.drop_duplicates(subset=['x'], inplace=True)

start_slice = 0
end_slice = edge_coord.shape[0] - 9
t_data = edge_coord.x_minute[start_slice:end_slice]
h_data = edge_coord.y_Mm[start_slice:end_slice] #different data slicing resulting in different result

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

t_onset = t_ons(tau_opt, c1_opt, c0_opt, t0_opt)
start_time = dt_intensity.loc[edge_coord.x.min()].date
onset_time = start_time + timedelta(minutes=t_onset)
print(f'onset time: {onset_time}')

t = t_data.apply(lambda i:start_time + timedelta(minutes=i))
dft_model = pd.Series(t_model)
model_t = dft_model.apply(lambda i:start_time + timedelta(minutes=i))

## utilize sympy
import sympy as smp
t,c0,tau,c1,c2,t0 = smp.symbols('t c0 tau c1 c2 t0', real=True)
h = c0 * smp.exp((t-t0)/tau) + c1*(t-t0) + c2

### Calculate Acceleration
d2hdt2 = smp.diff(h,t,2)
d2hdt2_f = smp.lambdify((t,c0,tau,c1,t0), d2hdt2)
ha = d2hdt2_f(t_model,c0=c0_opt,tau=tau_opt,c1=c1_opt,t0=t0_opt)
ha = ha *1000/60/60
# plt.plot(model_t, ha)
'''
def model_a(angular_separation, edge_coordinate, ref_line):
    ang_sep = angular_separation
    ang_sep['h_km'] = ang_sep.h_arcsec*one_arcsec_to_km
    edge_coord = edge_coordinate

    edge_coord['x_minute'] = edge_coord['x'].apply(lambda i: dt_intensity.loc[i].minute)
    edge_coord['x_second'] = edge_coord.x_minute*60
    edge_coord['y_Mm'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_km)/1000
    edge_coord['y_arcsec'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_arcsec)
    edge_coord.drop_duplicates(subset=['x'], inplace=True)

    start_slice = 0
    end_slice = 73
    t_data = edge_coord.x_minute[start_slice:end_slice]
    h_data = edge_coord.y_Mm[start_slice:end_slice] #different data slicing resulting in different result

    popt, pcov = curve_fit(model_h, t_data, h_data, p0=[7, 30, 0.1, 89, 211])
    # popt, pcov = curve_fit(model_h, t_data, h_data, p0=[0.001, 22, 0.15, 118, 242]) #p0=[355,128,0,336]
    c0_opt, tau_opt,c1_opt,c2_opt,t0_opt = popt
    t_model = np.linspace(min(t_data), max(t_data), int(max(t_data)-min(t_data)))
    h_model_mm = model_h(t_model,c0_opt,tau_opt,c1_opt,c2_opt, t0_opt)

    t_onset = t_ons(tau_opt, c1_opt, c0_opt, t0_opt) #purely from function, probably not present in dataset
    onset_t = t_model[t_model >= t_onset][0]  # always present in dataset
    start_time = dt_intensity.loc[edge_coord.x.min()].date
    onset_time = start_time + timedelta(minutes=onset_t)
    # print(f'onset time: {onset_time}')
    onset_height = edge_coord[edge_coord.x_minute >= onset_t].y_Mm.values[0]
    onset_height_arcsec = edge_coord[edge_coord.x_minute >= onset_t].y_arcsec.values[0]
    onset_h = h_model_mm[t_model >= onset_t][0]
    onset_h_arcsec = edge_coord[edge_coord.y_Mm>= onset_h].y_arcsec.values[0]

    t = t_data.apply(lambda i:start_time + timedelta(minutes=i))
    dft_model = pd.Series(t_model)
    model_t = dft_model.apply(lambda i:start_time + timedelta(minutes=i))

    ## utilize sympy
    import sympy as smp
    t,c0,tau,c1,c2,t0 = smp.symbols('t c0 tau c1 c2 t0', real=True)
    eq_h = c0 * smp.exp((t-t0)/tau) + c1*(t-t0) + c2
    dhdt = smp.diff(eq_h,t)
    # print(f'v(0) : {dhdt.subs([(t,8),(c0,c0_opt),(tau,tau_opt),(c1, c1_opt),(t0,t0_opt)]).evalf()}')
    dhdt_f = smp.lambdify((t,c0,tau,c1,t0), dhdt)
    hv = dhdt_f(t_model,c0=c0_opt,tau=tau_opt,c1=c1_opt,t0=t0_opt)
    hv = hv*1000/60 ## in km/s

    ### Calculate Acceleration
    d2hdt2 = smp.diff(eq_h,t,2)
    d2hdt2_f = smp.lambdify((t,c0,tau,c1,t0), d2hdt2)
    ha = d2hdt2_f(t_model,c0=c0_opt,tau=tau_opt,c1=c1_opt,t0=t0_opt)
    ha = ha *1000/60/60
    
    a_o = ha[model_t[model_t >= onset_time].index[0]] 
    
    return model_t, ha, start_time, onset_time, a_o, c0_opt, tau_opt,c1_opt,c2_opt,t0_opt, onset_t, onset_h, onset_h_arcsec, hv, ref_line
model_t, ha, start_time, onset_time, a_o, c0_opt, tau_opt,c1_opt,c2_opt,t0_opt, onset_t, onset_h, onset_h_arcsec, hv, ref_line = model_a(ang_sep_mid, edge_coord_mid, 'middle')
model_t_a, ha_a, start_time_a, onset_time_a, a_o_a, c0_opt_a, tau_opt_a,c1_opt_a,c2_opt_a,t0_opt_a, onset_t_a, onset_h_a, onset_h_arcsec_a, hv_a, ref_line_a = model_a(ang_sep_ab, edge_coord_ab, 'above')
model_t_b, ha_b, start_time_b, onset_time_b, a_o_b, c0_opt_b, tau_opt_b,c1_opt_b,c2_opt_b,t0_opt_b, onset_t_b, onset_h_b, onset_h_arcsec_b, hv_b, ref_line_b = model_a(ang_sep_be, edge_coord_be, 'below')

fig, ax = plt.subplots()
plt.rc('font', family='serif', serif='Times')
# plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)


# plt.plot(model_t, ha)
ax.plot(model_t,ha, color='k')
# ax.plot(model_t_a,ha_a, color='b')
# ax.plot(model_t_b,ha_b, color='g')
ax.set_ylim(0,0.05)
ax.set_xlim(start_time)
ax.set_ylabel('Accel. (km $s^-2$)', fontsize=18)
ax.set_xlabel('Start time = {}'.format(start_time.strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)

ax.text(dt_intensity.date[60],0.0452,'(f)',fontsize=20)

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.002))

# ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
ax.xaxis_date()
date_format = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=90))
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
ax.axvline(onset_time, color = 'black', ls = '-.')
# ax.axvline(onset_time_a, color = 'blue', ls = '-.')
# ax.axvline(onset_time_b, color = 'green', ls = '-.')
# ax.axvline(dt_intensity.date[200], color = 'black', ls = '--')

ax2 = ax.secondary_xaxis('top')
ax2.xaxis.set_tick_params(labeltop=False)
# ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=80))

ax3 = ax.secondary_yaxis('right')
ax3.yaxis.set_tick_params(labelright=False)
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.002))
# plt.axis('off') # for plain plot
# plt.savefig(f'Results/fitting_acceleration/fig_6f_all_lines_{dtdt}.png',bbox_inches='tight', dpi=100)
plt.show()

## because of ha shape is equal to model_t not edge_coord
## we using model_t instead of edge_coord
print(f'onset time: {onset_time}')
print(f'Acceleration at the onset point: {a_o*1000} m s^-2')
print(f'Final acceleration: {ha[-1]*1000} m s^-2')
print(f'onset time above: {onset_time_a}')
print(f'Acceleration at the onset point above: {a_o_a*1000} m s^-2')
print(f'Final acceleration above: {ha_a[-1]*1000} m s^-2')
print(f'onset time: {onset_time_b}')
print(f'Acceleration at the onset point: {a_o_b*1000} m s^-2')
print(f'Final acceleration: {ha_b[-1]*1000} m s^-2')
param_mid = np.array([[c0_opt,tau_opt, c1_opt, c2_opt,t0_opt, onset_time, onset_t, onset_h, onset_h_arcsec, hv[0], hv[-1], a_o*1000, ha[-1]*1000, ref_line]])
param_ab = np.array([[c0_opt_a,tau_opt_a, c1_opt_a, c2_opt_a,t0_opt_a, onset_time_a, onset_t_a, onset_h_a, onset_h_arcsec_a, hv_a[0], hv_a[-1], a_o_a*1000, ha_a[-1]*1000, ref_line_a]])
param_be = np.array([[c0_opt_b,tau_opt_b, c1_opt_b, c2_opt_b,t0_opt_b, onset_time_b, onset_t_b, onset_h_b, onset_h_arcsec_b, hv_b[0], hv_b[-1], a_o_b*1000, ha_b[-1]*1000, ref_line_b]])
param  = param_be
df_param = pd.DataFrame(param, columns=['c0', 'tau', 'c1', 'c2', 't0', 'onset_time', 'onset_t_minute', 'onset_h_mm', 'onset_h_arcsec', 'init_slow_v', 'max_v', 'onset_acc', 'end_acc', 'ref_line'], index=None)
# df_param.to_csv('Results/all/parameter.csv', mode='a', index=False, header=False)



