"""
This is for making fig 6b without fitting line and onset height information.
Skip this code and proceed to 1.3 won't affect anything.

If you wish to use latex-like font, add the following code on line-48. Careful! it will raise error on some machines.
    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)

Line 37-46 are used to improve the signal-to-noise ratio,adopted from blackbox.ai.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib
import os
import sunpy.map
import matplotlib.ticker as ticker
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

dtdt = '20120312'
ins = 'AIA304'
data_list = sorted(os.listdir(f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}'))
AIA_304 = f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}/{data_list[50]}'  ## careful on data_list[50]. Don't exceed the max number of data_list.
aia_map = sunpy.map.Map(AIA_304)             ## to ensure that sdoaia304 cmap has reloaded
sun_rad = aia_map.meta['RSUN_OBS']           ## in arcsec
sun_rad_ref = aia_map.meta['RSUN_REF']/1000  ## in km
one_arcsec_to_km = sun_rad_ref/sun_rad       ## conversion arcsec to km on Sun disk

intensity = np.load(f"Results/intensity_along_line/intensity_{dtdt}.npy")
dt_intensity = pd.read_csv(f"Results/intensity_along_line/datetime_intensity_{dtdt}.csv",parse_dates=['date'], date_format='%Y-%m-%dT%H:%M:%S.%f')
dt_intensity['minute'] = dt_intensity['date'].apply(lambda i:i - dt_intensity['date'][0])
dt_intensity['minute'] = dt_intensity['minute'].apply(lambda i:i.seconds/60)
ang_sep = pd.read_csv(f"Results/intensity_along_line/ang_sep_{dtdt}.csv")
ang_sep['h_km'] = ang_sep.h_arcsec*one_arcsec_to_km

bin_factor = int(np.round(intensity.shape[0] / 300))  # Calculate the bin factor
image_binned = np.mean(intensity.reshape(intensity.shape[0] // bin_factor, bin_factor, intensity.shape[1] // bin_factor, bin_factor), axis=(1, 3))
target_shape = (int(np.round(intensity.shape[0] / bin_factor / 2)) * 2, int(np.round(intensity.shape[1] / bin_factor / 2)) * 2)  # ~2" resolution
# Step 1: Multiply each row by its height
image_binned[:, 1] *= image_binned[:, 0]
# Step 2: Calculate the median value
median_value = np.median(image_binned)
# Step 3: Threshold the image above 1.5x the median value®
threshold_value = 0.01 * median_value
image_binned[image_binned < threshold_value] = np.nan
fig, ax = plt.subplots()

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=10)

orig_map=matplotlib.colormaps.get_cmap('sdoaia304')
reversed_map = orig_map.reversed()
xlim = [dt_intensity.date[0],dt_intensity.date[len(dt_intensity)-1]]
x_lim = mdates.date2num(xlim)
ax.imshow(image_binned, cmap=reversed_map, origin='lower', vmax=35, extent=[x_lim[0],x_lim[1],0,ang_sep.h_arcsec.max()], aspect='auto')   ## aspect='auto' keeps image on landscape.
ax.set_ylabel('Distance along slice (arcsec)', fontsize=18)
ax.set_xlabel('Start time = {}'.format(dt_intensity.date[0].strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)

ax.text(dt_intensity.date[10],450,'(b)',fontsize=20)   ## adjust this! different dataset, different xy-number

ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
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
os.makedirs('Results/height-time_plot', exist_ok=True)
plt.savefig(f'Results/height-time_plot/fig_6b_lite_{dtdt}.png',bbox_inches='tight', dpi=100)
plt.show()
