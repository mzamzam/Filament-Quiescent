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
AIA_304 = f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}/{data_list[200]}'
aia_map = sunpy.map.Map(AIA_304)
sun_rad = aia_map.meta['RSUN_OBS'] #in arcsec
sun_rad_ref = aia_map.meta['RSUN_REF']/1000 #in km
one_arcsec_to_km = sun_rad_ref/sun_rad
scale = aia_map.scale[0].value
def eq_h(t,c0,tau,c1,c2,t0):
    return c0 * np.exp((t-t0)/ tau) + c1 * (t-t0) + c2
def t_ons(tau,c1,c0,t0):
    return tau*np.log(c1*tau/c0)+t0

t,c0,tau,c1,c2,t0 = smp.symbols('t c0 tau c1 c2 t0', real=True)
h = c0 * smp.exp((t-t0)/tau) + c1*(t-t0) + c2
dhdt = smp.diff(h,t)   ## differentiate one time to get velocity
dhdt_f = smp.lambdify((t,c0,tau,c1,t0), dhdt)

ang_sep_mid = pd.read_csv(f"Results/intensity_along_line/ang_sep_{dtdt}.csv")
dt_intensity = pd.read_csv(f"Results/intensity_along_line/datetime_intensity_{dtdt}.csv",parse_dates=['date'], date_format='%Y-%m-%dT%H:%M:%S.%f')
dt_intensity['minute'] = dt_intensity['date'].apply(lambda i:i - dt_intensity['date'][0])
dt_intensity['minute'] = dt_intensity['minute'].apply(lambda i:i.seconds/60)

edge_coord_mid = pd.read_csv(f"Results/canny/edge_coord_{dtdt}.csv", index_col='Unnamed: 0')
def model_v(angular_separation, edge_coordinate):
    ang_sep = angular_separation
    ang_sep['h_km'] = ang_sep.h_arcsec*one_arcsec_to_km
    edge_coord = edge_coordinate

    edge_coord['x_minute'] = edge_coord['x'].apply(lambda i: dt_intensity.loc[i].minute)
    edge_coord['x_second'] = edge_coord.x_minute*60
    edge_coord['y_Mm'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_km)/1000
    edge_coord['y_Km'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_km)
    edge_coord['y_arcsec'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_arcsec)
    edge_coord.drop_duplicates(subset=['x'], inplace=True)

    start_slice = 0
    end_slice = 70   ## or can be used: edge_coord.shape[0]
    t_data = edge_coord.x_minute[start_slice:end_slice]
    h_data = edge_coord.y_Mm[start_slice:end_slice] #different data slicing resulting in different result

    popt, pcov = curve_fit(eq_h, t_data, h_data, p0=[7, 30, 0.1, 89, 211])
    c0_opt, tau_opt,c1_opt,c2_opt,t0_opt = popt
    t_model = np.linspace(min(t_data), max(t_data), int(max(t_data)-min(t_data)))
    h_model = eq_h(t_model,c0_opt,tau_opt,c1_opt,c2_opt, t0_opt)
    t_onset = t_ons(tau_opt, c1_opt, c0_opt, t0_opt)
    start_time = dt_intensity.loc[1].date
    onset_time = start_time + timedelta(minutes=t_onset)

    ti = t_data.apply(lambda i:start_time + timedelta(minutes=i))
    dft_model = pd.Series(t_model)
    model_t = dft_model.apply(lambda i:start_time + timedelta(minutes=i))

    hv = dhdt_f(t_model,c0=c0_opt,tau=tau_opt,c1=c1_opt,t0=t0_opt)
    hv = hv*1000/60 ## in km/s
    
    return start_time, model_t, hv, onset_time

start_time, model_t, hv, onset_time = model_v(ang_sep_mid, edge_coord_mid)

print(f'onset time: {onset_time}')
print(f'Initial slow rise velocity: {hv[0]} km s-1')
print(f'Maximum velocity in the AIA FOV: {hv[-1]} km s-1')

fig, ax = plt.subplots()
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)

ax.plot(model_t,hv, color='k')

ax.set_xlim(start_time)
ax.set_ylabel('Velocity (km $s^-1$)', fontsize=18)
ax.set_xlabel('Start time = {}'.format(start_time.strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)
ax.text(dt_intensity.date[7],110,'(e)',fontsize=20)

ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

ax.xaxis_date()
date_format = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=90))
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
ax.axvline(onset_time, color = 'black', ls = 'dotted')

ax2 = ax.secondary_xaxis('top')
ax2.xaxis.set_tick_params(labeltop=False)
ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=80))

ax3 = ax.secondary_yaxis('right')
ax3.yaxis.set_tick_params(labelright=False)
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(5))
os.makedirs('Results/fitting_velocity/', exist_ok=True)
plt.savefig(f'Results/fitting_velocity/fig_6e_{dtdt}.png',bbox_inches='tight', dpi=100)
plt.show()




