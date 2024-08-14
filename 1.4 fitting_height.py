"""
You can use chat-gpt to find p0 on line-50
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
edge_coord_mid = pd.read_csv(f"Results/canny/edge_coord_{dtdt}.csv", index_col='Unnamed: 0')
def model_ht(angular_separation, edge_coordinate):
    ang_sep = angular_separation
    ang_sep['h_km'] = ang_sep.h_arcsec*one_arcsec_to_km
    edge_coord = edge_coordinate
    edge_coord['x_minute'] = edge_coord['x'].apply(lambda i: dt_intensity.loc[i].minute)
    edge_coord['x_second'] = edge_coord.x_minute*60
    edge_coord['y_Mm'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_km)/1000
    edge_coord['y_arcsec'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_arcsec)
    edge_coord.drop_duplicates(subset=['x'], inplace=True)
    start_time = dt_intensity.loc[1].date

    start_slice = 0
    end_slice = 70   ## or can be used: edge_coord.shape[0]
    t_data = edge_coord.x_minute[start_slice:end_slice]
    h_data = edge_coord.y_Mm[start_slice:end_slice]   ## different data slicing resulting in different result

    popt, pcov = curve_fit(eq_h, t_data, h_data, p0=[7, 30, 0.1, 89, 211])   ## guessing p0 is tricky, need alot of trials and errors.
    c0_opt, tau_opt,c1_opt,c2_opt,t0_opt = popt
    t_model = np.linspace(min(t_data), max(t_data), int(max(t_data)-min(t_data)))
    h_model = eq_h(t_model,c0_opt,tau_opt,c1_opt,c2_opt, t0_opt)
    dft_model = pd.Series(t_model)
    model_t = dft_model.apply(lambda i:start_time + timedelta(minutes=i))
    t = t_data.apply(lambda i:start_time + timedelta(minutes=i))

    t_onset = t_ons(tau_opt, c1_opt, c0_opt, t0_opt)
    onset_t = t_model[t_model >= t_onset][0]
    onset_time = start_time + timedelta(minutes=onset_t)
    onset_height = edge_coord[edge_coord.x_minute >= onset_t].y_Mm.values[0]
    onset_height_arcsec = edge_coord[edge_coord.x_minute >= onset_t].y_arcsec.values[0]
    onset_h = h_model[np.argwhere(t_model == onset_t)[0][0]]
    onset_h_arcsec = edge_coord[edge_coord.y_Mm >= onset_h].y_arcsec.values[0]
    return c0_opt, tau_opt, c1_opt, c2_opt, t0_opt, t_onset, onset_time, onset_h, onset_h_arcsec, model_t, h_model, start_time, t_data, h_data
    
c0_opt, tau_opt, c1_opt, c2_opt, t0_opt, t_onset, onset_time, onset_h, onset_h_arcsec, model_t, h_model, start_time, t_data, h_data = model_ht(ang_sep_mid, edge_coord_mid)

print('onset time: {} UT'.format(onset_time.strftime('%Y/%m/%d %H:%M')))
print(f'onset height: {onset_h} Mm')

fig, ax = plt.subplots()
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=12)

ax.plot(model_t,h_model,'k')
ax.set_xlim(start_time)
ax.set_ylabel('Height (Mm)', fontsize=18)
ax.set_xlabel('Start time = {}'.format(start_time.strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)

ax.text(dt_intensity.date[9],320,'(d)',fontsize=20)

ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

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
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(10))
os.makedirs('Results/fitting_height', exist_ok=True)
plt.savefig(f'Results/fitting_height/fig_6d_{dtdt}.png',bbox_inches='tight', dpi=100)
plt.show()



