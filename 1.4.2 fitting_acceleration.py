import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import sunpy.map
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import sympy as smp

dtdt = '20120312'
ins = 'AIA304'
data_list = sorted(os.listdir(f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}'))
AIA_304 = f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}/{data_list[50]}'
aia_map = sunpy.map.Map(AIA_304)
sun_rad = aia_map.meta['RSUN_OBS']          ## in arcsec
sun_rad_ref = aia_map.meta['RSUN_REF']/1000 ## in km
one_arcsec_to_km = sun_rad_ref/sun_rad
C2_height = 2.5 * sun_rad_ref/1000        ## in Mm
cme_t0 = pd.to_datetime('2012/03/12 01:25') ## from cactus
cme_v = 422                                 ## km/s from cactus
def eq_h(t,c0,tau,c1,c2,t0):
    return c0 * np.exp((t-t0)/ tau) + c1 * (t-t0) + c2
def t_ons(tau,c1,c0,t0):
    return tau*np.log(c1*tau/c0)+t0
t,c0,tau,c1,c2,t0 = smp.symbols('t c0 tau c1 c2 t0', real=True)
h = c0 * smp.exp((t-t0)/tau) + c1*(t-t0) + c2
dhdt = smp.diff(h,t)
dhdt_f = smp.lambdify((t,c0,tau,c1,t0), dhdt)
d2hdt2 = smp.diff(h,t,2)
d2hdt2_f = smp.lambdify((t,c0,tau,c1,t0), d2hdt2)

ang_sep_mid = pd.read_csv(f"Results/intensity_along_line/ang_sep_{dtdt}.csv")
dt_intensity = pd.read_csv(f"Results/intensity_along_line/datetime_intensity_{dtdt}.csv",parse_dates=['date'], date_format='%Y-%m-%dT%H:%M:%S.%f')
dt_intensity['minute'] = dt_intensity['date'].apply(lambda i:i - dt_intensity['date'][0])
dt_intensity['minute'] = dt_intensity['minute'].apply(lambda i:i.seconds/60)
start_time = dt_intensity.loc[1].date
end_time = dt_intensity.loc[dt_intensity.shape[0]-1].date
edge_coord_mid = pd.read_csv(f"Results/canny/edge_coord_{dtdt}.csv", index_col='Unnamed: 0')
def model_a(angular_separation, edge_coordinate, ref_line):
    ang_sep = angular_separation
    ang_sep['h_km'] = ang_sep.h_arcsec*round(one_arcsec_to_km)
    edge_coord = edge_coordinate
    edge_coord['x_second'] = edge_coord.x_minute*60
    edge_coord['y_Mm'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_km)/1000
    edge_coord.drop_duplicates(subset=['x'], inplace=True)

    start_slice = 0
    end_slice = 70
    t_minute = edge_coord.x_minute[start_slice:end_slice]
    h_Mm = edge_coord.y_Mm[start_slice:end_slice] #different data slicing resulting in different result

    popt, pcov = curve_fit(eq_h, t_minute, h_Mm, p0=[7, 30, 0.1, 89, 211])
    c0_opt, tau_opt,c1_opt,c2_opt,t0_opt = popt
    t_minute_model = np.linspace(min(t_minute), max(t_minute), int(max(t_minute)-min(t_minute)))
    h_Mm_model = eq_h(t_minute_model,c0_opt,tau_opt,c1_opt,c2_opt, t0_opt)
    t_minute_dt = t_minute.apply(lambda i: start_time + timedelta(minutes=i))
    dft_model = pd.Series(t_minute_model)
    t_minute_model_dt = dft_model.apply(lambda i: start_time + timedelta(minutes=i))

    t_minute_onset = t_ons(tau_opt, c1_opt, c0_opt, t0_opt) ## purely from function, probably not present in dataset
    t_minute_onset_model = t_minute_model[t_minute_model >= t_minute_onset][0]         ## always present in dataset
    t_onset_datetime = start_time + timedelta(minutes=t_minute_onset_model)
    h_onset_Mm_data = edge_coord[edge_coord.x_minute >= t_minute_onset_model].y_Mm.values[0]
    h_onset_arcsec_datax = edge_coord[edge_coord.x_minute >= t_minute_onset_model].y_arcsec.values[0]
    h_onset_Mm_model = h_Mm_model[t_minute_model >= t_minute_onset_model][0]
    h_onset_arcsec_datay = edge_coord[edge_coord.y_Mm>= h_onset_Mm_model].y_arcsec.values[0]

    hv = dhdt_f(t_minute_model,c0=c0_opt,tau=tau_opt,c1=c1_opt,t0=t0_opt)
    hv = hv*1000/60 ## in km/s

    ha = d2hdt2_f(t_minute_model,c0=c0_opt,tau=tau_opt,c1=c1_opt,t0=t0_opt)
    ha = ha *1000/60/60
    a_o = ha[t_minute_model_dt[t_minute_model_dt >= t_onset_datetime].index[0]]
    ## because of ha shape is equal to model_t not edge_coord
    ## we using model_t instead of edge_coord
    return start_time, t_minute_model_dt, hv, t_onset_datetime, h_onset_Mm_model,t_minute_onset_model, h_onset_arcsec_datay, a_o, ha, c0_opt, tau_opt,c1_opt,c2_opt,t0_opt, ref_line
start_time, t_minute_model_dt, hv, t_onset_datetime, h_onset_Mm_model, t_minute_onset_model, h_onset_arcsec_datay, a_o, ha, c0_opt, tau_opt,c1_opt,c2_opt,t0_opt, ref_line = model_a(ang_sep_mid, edge_coord_mid, 'middle')
print(f'optimum parameter (c0, tau, c1, c2, t0): {round(c0_opt,3), round(tau_opt,3), round(c1_opt,3), round(c2_opt,3), round(t0_opt,3)}')
print('onset time: {} UT'.format(t_onset_datetime.strftime('%Y/%m/%d %H:%M')))
print(f'onset height: {h_onset_Mm_model} Mm')
print(f'Initial slow rise velocity: {hv[0]} km s-1')
print(f'Maximum velocity in the AIA FOV: {hv[-1]} km s-1')
print(f'Acceleration at the onset point: {a_o*1000} m s^-2')
print(f'Final acceleration: {ha[-1]*1000} m s^-2')
def eq_t(t_atC2):        ## from chatGPT
    return c0_opt * np.exp((t_atC2-t0_opt)/ tau_opt) + c1_opt * (t_atC2-t0_opt) + c2_opt - C2_height
t_atC2_guess = t0_opt    ## t0_opt as a reasonable guess
from scipy.optimize import fsolve
t_atC2_solution = fsolve(eq_t, t_atC2_guess)
t_atC2_solution_dt = t_onset_datetime + timedelta(minutes=t_atC2_solution[0])
hv_atC2 = dhdt_f(t_atC2_solution[0],c0=c0_opt,tau=tau_opt,c1=c1_opt,t0=t0_opt)
hv_atC2 = hv_atC2*1000/60 ## in km/s
print(f'Arrive at the C2 FOV at {t_atC2_solution_dt} UT with a velocity of {hv_atC2} km/s')
print(f'CME onset (t0) from Cactus is {cme_t0} UT with a velocity {cme_v} km/s')

param_mid = np.array([[c0_opt,tau_opt, c1_opt, c2_opt,t0_opt, t_onset_datetime, t_minute_onset_model, h_onset_Mm_model, h_onset_arcsec_datay, hv[0], hv[-1], a_o*1000, ha[-1]*1000, t_atC2_solution_dt , cme_t0, hv_atC2, cme_v, ref_line]])
param = param_mid
df_param = pd.DataFrame(param, columns=['c0', 'tau', 'c1', 'c2', 't0', 'onset_time', 'onset_time_model', 'onset_h_Mm_model', 'onset_h_arcsec', 'init_slow_v', 'max_v', 'onset_acc', 'end_acc', 't_atC2', 't_atC2_cactus', 'v_atC2', 'v_atC2_cactus', 'ref_line'], index=None)
df_param.to_csv(f'Results/parameter_{dtdt}.csv', index=False, header=True)

fig, ax = plt.subplots()
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)

ax.plot(t_minute_model_dt,ha, color='k')
ax.set_ylim(0, 0.05)
ax.set_xlim(start_time, end_time)
ax.set_ylabel('Accel. (km $s^-2$)', fontsize=18)
ax.set_xlabel('Start time = {}'.format(start_time.strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)

ax.text(dt_intensity.date[10],0.040,'(f)',fontsize=20)

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.002))

ax.xaxis_date()
date_format = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=90))
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
ax.axvline(t_onset_datetime, color = 'black', ls = 'dotted')

ax2 = ax.secondary_xaxis('top')
ax2.xaxis.set_tick_params(labeltop=False)
ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=80))

ax3 = ax.secondary_yaxis('right')
ax3.yaxis.set_tick_params(labelright=False)
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.002))
os.makedirs('Results/fitting_acceleration', exist_ok=True)
plt.savefig(f'Results/fitting_acceleration/fig_6f_{dtdt}.png',bbox_inches='tight', dpi=100)
plt.show()

