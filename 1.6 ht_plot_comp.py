'''
For making plot 6b with fitting line and onset height mark

We have fitting line in Mm, but we need to plot in arcsec.
See line 97-98 for the conversion.
'''
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

dtdt = '20120312'
ins = 'AIA304'
data_list = sorted(os.listdir(f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}'))
AIA_304 = f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}/{data_list[50]}'
aia_map = sunpy.map.Map(AIA_304) #to ensure that sdoaia304 cmap has reloaded
sun_rad = aia_map.meta['RSUN_OBS'] #in arcsec
sun_rad_ref = aia_map.meta['RSUN_REF']/1000 #in km
one_arcsec_to_km = sun_rad_ref/sun_rad

intensity = np.load(f"Results/intensity_along_line/intensity_{dtdt}.npy")
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

def eq_h(t, c0, tau, c1, c2, t0):
    return c0 * np.exp((t-t0) / tau) + c1 * (t-t0) + c2
def t_ons(tau, c1, c0, t0):
    return tau*np.log(c1*tau/c0)+t0
ang_sep_mid = pd.read_csv(f"Results/intensity_along_line/ang_sep_{dtdt}.csv")
dt_intensity = pd.read_csv(f"Results/intensity_along_line/datetime_intensity_{dtdt}.csv", parse_dates=['date'], date_format='%Y-%m-%dT%H:%M:%S.%f')
dt_intensity['minute'] = dt_intensity['date'].apply(lambda i:i - dt_intensity['date'][0])
dt_intensity['minute'] = dt_intensity['minute'].apply(lambda i:i.seconds/60)
start_time = dt_intensity.loc[1].date
end_time = dt_intensity.loc[dt_intensity.shape[0]-1].date
edge_coord_mid = pd.read_csv(f"Results/canny/edge_coord_{dtdt}.csv", index_col='Unnamed: 0')

def model_ht(angular_separation, edge_coordinate):
    ang_sep = angular_separation
    ang_sep['h_km'] = ang_sep.h_arcsec*round(one_arcsec_to_km)   ## not rounded, error raise easily
    edge_coord = edge_coordinate
    edge_coord['x_second'] = edge_coord.x_minute*60
    edge_coord['y_Mm'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_km)/1000
    edge_coord.drop_duplicates(subset=['x'], inplace=True)

    start_slice = 0
    end_slice = 70   ## or can be used: edge_coord.shape[0]
    t_minute = edge_coord.x_minute[start_slice:end_slice]
    h_Mm = edge_coord.y_Mm[start_slice:end_slice]   ## different data slicing resulting in different result
    h_arcsec = edge_coord.y_arcsec[start_slice:end_slice]

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
    return c0_opt, tau_opt, c1_opt, c2_opt, t0_opt, t_minute_onset, t_onset_datetime, h_onset_Mm_model, h_onset_arcsec_datay, t_minute_model_dt, h_Mm_model, start_time, t_minute, t_minute_dt, h_Mm, h_arcsec
c0_opt, tau_opt, c1_opt, c2_opt, t0_opt, t_minute_onset, t_onset_datetime, h_onset_Mm_model, h_onset_arcsec_datay, t_minute_model_dt, h_Mm_model, start_time, t_minute, t_minute_dt, h_Mm, h_arcsec = model_ht(ang_sep_mid, edge_coord_mid)
print(f'optimum parameter (c0, tau, c1, c2, t0): {round(c0_opt,3), round(tau_opt,3), round(c1_opt,3), round(c2_opt,3), round(t0_opt,3)}')
print('onset time: {} UT'.format(t_onset_datetime.strftime('%Y/%m/%d %H:%M')))
print(f'onset height: {h_onset_Mm_model} Mm')

fig, ax = plt.subplots()
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)

orig_map=matplotlib.colormaps.get_cmap('sdoaia304')
reversed_map = orig_map.reversed()
xlim = [dt_intensity.date[0],dt_intensity.date[len(dt_intensity)-1]]
x_lim = mdates.date2num(xlim)
ax.imshow(image_binned, cmap=reversed_map, origin='lower', vmax=35, extent=[x_lim[0],x_lim[1],0,ang_sep_mid.h_arcsec.max()], aspect='auto')
one_km_to_arcsec = 1/one_arcsec_to_km
h_arcsec_model = (h_Mm_model*1000)*one_km_to_arcsec            ## only have fitting line in Mm, need to convert the to arcsec
ax.plot(t_minute_model_dt, h_arcsec_model, color='k')    ## plot fitting line but in arcsec
ax.set_ylabel('Distance along slice (arcsec)', fontsize=18)
ax.set_xlabel('Start time = {}'.format(start_time.strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)
ax.set_ylim(0,470)
ax.text(dt_intensity.date[10],420,'(b)',fontsize=20)
ax.text(t_onset_datetime-timedelta(minutes=45),h_onset_arcsec_datay+13,f'{round(h_onset_Mm_model)} Mm',fontsize=13)
ax.plot(t_onset_datetime,h_onset_arcsec_datay, 'k*', ms=10)

ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
# ax.xaxis_date()
date_format = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=90))
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)

ax2 = ax.secondary_xaxis('top')
ax2.xaxis.set_tick_params(labeltop=False)
ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=80))

ax3 = ax.secondary_yaxis('right')
ax3.yaxis.set_tick_params(labelright=False)
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(10))
os.makedirs('Results/ht_plot_comp', exist_ok=True)
plt.savefig(f'Results/ht_plot_comp/fig_6b_{dtdt}.png',bbox_inches='tight', dpi=100)
plt.show()

