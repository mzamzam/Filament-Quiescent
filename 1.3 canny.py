import numpy as np
from matplotlib import pyplot as plt
from skimage import feature
import os
import sunpy.map
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import timedelta

dtdt = '20120312'
ins = 'AIA304'
data_list = sorted(os.listdir(f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}'))
AIA_304 = f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}/{data_list[50]}'
aia_map = sunpy.map.Map(AIA_304)
sun_rad = aia_map.meta['RSUN_OBS']
sun_rad_ref = aia_map.meta['RSUN_REF']/1000
one_arcsec_to_km = sun_rad_ref/sun_rad

intensity_mid = np.load(f"Results/intensity_along_line/intensity_{dtdt}.npy")
dt_intensity = pd.read_csv(f"Results/intensity_along_line/datetime_intensity_{dtdt}.csv",parse_dates=['date'], date_format='%Y-%m-%dT%H:%M:%S.%f')
dt_intensity['minute'] = dt_intensity['date'].apply(lambda i:i - dt_intensity['date'][0])
dt_intensity['minute'] = dt_intensity['minute'].apply(lambda i:i.seconds/60)
ang_sep_mid = pd.read_csv(f"Results/intensity_along_line/ang_sep_{dtdt}.csv")
intensity = intensity_mid
ang_sep = ang_sep_mid
ang_sep['h_km'] = ang_sep.h_arcsec*one_arcsec_to_km
orig_map = plt.colormaps.get('sdoaia304')
reversed_map = orig_map.reversed()
sig = 9
arr = feature.canny(intensity, sigma=sig)

fig, ax = plt.subplots()
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)

coord = np.argwhere(arr==1)   ## detect all edges from array numpy
df_coord = pd.DataFrame(coord, columns=['y', 'x'])
df_coord['y_arcsec'] = df_coord['y'].apply(lambda i:ang_sep.loc[i].h_arcsec)
df_coord['x_minute'] = df_coord['x'].apply(lambda i: dt_intensity.loc[i].minute)

df_coord = df_coord.sort_values(by=['x_minute'])
df_coord_edge = df_coord[df_coord.y_arcsec > (df_coord.y_arcsec[:5].max() - 1)]
df_coord_edge.drop_duplicates(subset=['y_arcsec'], inplace=True)
os.makedirs('Results/canny', exist_ok=True)
df_coord_edge = df_coord_edge[df_coord_edge.x_minute >= 120]
df_coord_edge.to_csv(f'Results/canny/edge_coord_{dtdt}_{sig}.csv')

start_time = dt_intensity.loc[df_coord.x.min()].date
coord_time = df_coord.x_minute.apply(lambda i:start_time + timedelta(minutes=i))
coord_edge_time = df_coord_edge.x_minute.apply(lambda i:start_time + timedelta(minutes=i))
ax.plot(coord_time, df_coord.y_arcsec, color='cyan', marker='.', linestyle='None', markersize=2)
ax.plot(coord_edge_time, df_coord_edge.y_arcsec, color='red', marker='.', linestyle='None', markersize=2)
ax.set_ylabel('Height (arcsec)', fontsize=18)
ax.set_xlabel('Start time = {}'.format(start_time.strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)

ax.text(dt_intensity.date[7],450,'(c)',fontsize=20)

ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
date_format = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=90))
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
# ax.axvline(dt_intensity.date[190],0,1, color = 'black', ls = '--')   ## if want to add vertical line on the graph, uncomment this line.

ax2 = ax.secondary_xaxis('top')
ax2.xaxis.set_tick_params(labeltop=False)
ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=80))

ax3 = ax.secondary_yaxis('right')
ax3.yaxis.set_tick_params(labelright=False)
ax3.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax3.yaxis.set_minor_locator(ticker.MultipleLocator(20))
plt.tight_layout()
plt.savefig(f'Results/canny/fig_6c_{dtdt}_{sig}.png', dpi=200)
plt.show()



