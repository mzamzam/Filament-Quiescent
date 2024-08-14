"""
You can use chat-gpt to find p0 (line-47)
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

dtdt = '20120312'
ins = 'AIA304'
data_list = sorted(os.listdir(f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}'))
AIA_304 = f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}/{data_list[200]}'
aia_map = sunpy.map.Map(AIA_304)
sun_rad = aia_map.meta['RSUN_OBS']   ## in arcsec
sun_rad_ref = aia_map.meta['RSUN_REF']/1000   ## in km
one_arcsec_to_km = sun_rad_ref/sun_rad
def eq_h(t,c0,tau,c1,c2,t0):
    ## this is equation (1) from Su et al (2015)
    return c0 * np.exp((t-t0)/ tau) + c1 * (t-t0) + c2
def t_ons(tau,c1,c0,t0):
    ## this is equation (2) from Su et al (2015)
    return tau*np.log(c1*tau/c0)+t0
ang_sep_mid = pd.read_csv(f"Results/intensity_along_line/ang_sep_{dtdt}.csv")
dt_intensity = pd.read_csv(f"Results/intensity_along_line/datetime_intensity_{dtdt}.csv",parse_dates=['date'], date_format='%Y-%m-%dT%H:%M:%S.%f')
dt_intensity['minute'] = dt_intensity['date'].apply(lambda i:i - dt_intensity['date'][0])
dt_intensity['minute'] = dt_intensity['minute'].apply(lambda i:i.seconds/60)
start_time = dt_intensity.loc[1].date
edge_coord_mid = pd.read_csv(f"Results/canny/edge_coord_{dtdt}.csv", index_col='Unnamed: 0')
def model_ht(angular_separation, edge_coordinate):
    ang_sep = angular_separation
    ang_sep['h_km'] = ang_sep.h_arcsec*one_arcsec_to_km
    edge_coord = edge_coordinate
    edge_coord['x_second'] = edge_coord.x_minute*60
    edge_coord['y_Mm'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_km)/1000
    edge_coord.drop_duplicates(subset=['x'], inplace=True)

    start_slice = 0
    end_slice = 70   ## or can be used: edge_coord.shape[0]
    t_minute = edge_coord.x_minute[start_slice:end_slice]
    h_Mm = edge_coord.y_Mm[start_slice:end_slice]   ## different data slicing resulting in different result

    popt, pcov = curve_fit(eq_h, t_minute, h_Mm, p0=[7, 30, 0.1, 89, 211])   ## guessing p0 is tricky, need alot of trials and errors.
    c0_opt, tau_opt,c1_opt,c2_opt,t0_opt = popt
    t_minute_model = np.linspace(min(t_minute), max(t_minute), int(max(t_minute)-min(t_minute)))
    h_Mm_model = eq_h(t_minute_model,c0_opt,tau_opt,c1_opt,c2_opt, t0_opt)
    dft_model = pd.Series(t_minute_model)
    t_minute_model_dt = dft_model.apply(lambda i:start_time + timedelta(minutes=i))
    t_minute_dt = t_minute.apply(lambda i:start_time + timedelta(minutes=i))

    t_minute_onset = t_ons(tau_opt, c1_opt, c0_opt, t0_opt)
    t_minute_onset_model = t_minute_model[t_minute_model >= t_minute_onset][0]
    t_onset_datetime = start_time + timedelta(minutes=t_minute_onset_model)
    h_onset_Mm_data = edge_coord[edge_coord.x_minute >= t_minute_onset_model].y_Mm.values[0]
    h_onset_arcsec_datax = edge_coord[edge_coord.x_minute >= t_minute_onset_model].y_arcsec.values[0]
    h_onset_Mm_model = h_Mm_model[np.argwhere(t_minute_model == t_minute_onset_model)[0][0]]
    h_onset_arcsec_datay = edge_coord[edge_coord.y_Mm >= h_onset_Mm_model].y_arcsec.values[0]
    return c0_opt, tau_opt, c1_opt, c2_opt, t0_opt, t_minute_onset, t_onset_datetime, h_onset_Mm_model, h_onset_arcsec_datay, t_minute_model_dt, h_Mm_model, start_time, t_minute, h_Mm
    
c0_opt, tau_opt, c1_opt, c2_opt, t0_opt, t_minute_onset, t_onset_datetime, h_onset_Mm_model, h_onset_arcsec_datay, t_minute_model_dt, h_Mm_model, start_time, t_minute, h_Mm = model_ht(ang_sep_mid, edge_coord_mid)

print('onset time: {} UT'.format(t_onset_datetime.strftime('%Y/%m/%d %H:%M')))
print(f'onset height: {h_onset_Mm_model} Mm')

fig, ax = plt.subplots()
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=12)

ax.plot(t_minute_model_dt,h_Mm_model,'k')
ax.set_ylim(0,350)
ax.set_xlim(start_time)
ax.set_ylabel('Height (Mm)', fontsize=18)
ax.set_xlabel('Start time = {}'.format(start_time.strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)

ax.text(dt_intensity.date[9],320,'(d)',fontsize=20)

ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

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
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(10))
os.makedirs('Results/fitting_height', exist_ok=True)
plt.savefig(f'Results/fitting_height/fig_6d_{dtdt}.png',bbox_inches='tight', dpi=100)



