# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:38:18 2024

@author: WIN 11
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import sunpy.map
import astropy.units as u
import matplotlib
from astropy.coordinates import SkyCoord
import matplotlib.ticker as ticker
from skimage import feature
from datetime import timedelta
from scipy.optimize import curve_fit
import sympy as smp
dtdt = '20120312'
# dtdt = '20230420'
ins = 'AIA304'
data_list = os.listdir(f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}')
df = pd.DataFrame()
ht = np.zeros([307,len(data_list)]) #307 is an array size of black line.
ht_above = np.zeros([318,len(data_list)]) #315 is an array size of blue line
ht_below = np.zeros([297,len(data_list)]) #299 is an array of green line
## number of array should determined manually.
## you cannot assume all lines have the same array. It will raise an error.
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
orig_map=matplotlib.colormaps.get_cmap('sdoaia304')
reversed_map = orig_map.reversed()
save_fd = 'Results/all'
os.makedirs(f'{save_fd}', exist_ok=True)

example_case = 174

in_fd = "Results/intensity_along_line"
os.makedirs(f'{in_fd}', exist_ok=True)
def plot_6a(i, x): #type x='save_all' to save all figures, x = 'save_one' to save only one figure, otherwise won't save figure
    '''
    This function is modified version of 1.1 intensity_along_line.py
    
    '''
    AIA_304 = f'Data/AIA304/2012/03/{data_list[i]}'
    aia_map = sunpy.map.Map(AIA_304)
    
    xlims_world = [-900, -380]*u.arcsec  #determined manually
    ylims_world = [-1200, -630]*u.arcsec #determined manually
    world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world, frame=aia_map.coordinate_frame)
    pixel_coords_x, pixel_coords_y = aia_map.wcs.world_to_pixel(world_coords)
    
    x_start, y_start = -500, -840 #coordinate for the initial line at the solar surface
    x_end, y_end = -875, -1200 #coordinate for the end line 
    xe_above, ye_above = -902, -1200 #it should be offset by 2 degree
    xe_below, ye_below = -850, -1200 #it should be offset by 2 degree#it should be offset by 2 degree
    
    line_coords = SkyCoord([x_start, x_end], [y_start, y_end], unit=(u.arcsec, u.arcsec),
                           frame=aia_map.coordinate_frame)
    intensity_coords = sunpy.map.pixelate_coord_path(aia_map, line_coords)
    intensity = sunpy.map.sample_at_coords(aia_map, intensity_coords)
    angular_separation = intensity_coords.separation(intensity_coords[0]).to(u.arcsec)
    
    line_coords_above = SkyCoord([x_start, xe_above], [y_start, ye_above], unit=(u.arcsec, u.arcsec),
                           frame=aia_map.coordinate_frame)
    intensity_coords_above = sunpy.map.pixelate_coord_path(aia_map, line_coords_above)
    intensity_above = sunpy.map.sample_at_coords(aia_map, intensity_coords_above)
    angular_separation_above = intensity_coords_above.separation(intensity_coords_above[0]).to(u.arcsec)
    
    line_coords_below = SkyCoord([x_start, xe_below], [y_start, ye_below], unit=(u.arcsec, u.arcsec),
                           frame=aia_map.coordinate_frame)
    intensity_coords_below = sunpy.map.pixelate_coord_path(aia_map, line_coords_below)
    intensity_below = sunpy.map.sample_at_coords(aia_map, intensity_coords_below)
    angular_separation_below = intensity_coords_below.separation(intensity_coords_below[0]).to(u.arcsec)
    
    data_date = aia_map.date.value
    df.loc[i, 'date'] = data_date
    
    ht[:,i] = intensity
    ht_above[:,i] = intensity_above
    ht_below[:,i] = intensity_below
    
    # orig_map=matplotlib.colormaps.get_cmap('sdoaia304')
    # reversed_map = orig_map.reversed()
    fig = plt.figure()
    ax1 = fig.add_subplot(projection=aia_map)
    aia_map.plot(axes=ax1, cmap=reversed_map, clip_interval=(1, 99.9)*u.percent)
    ax1.set_xlim(pixel_coords_x)
    ax1.set_ylim(pixel_coords_y)
    ax1.plot_coord(intensity_coords, lw=1, color='black')
    ax1.plot_coord(intensity_coords_above, lw=1, color='blue')
    ax1.plot_coord(intensity_coords_below, lw=1, color='green')
    ax1.plot_coord(SkyCoord(-845*u.arcsec, -700*u.arcsec, frame=aia_map.coordinate_frame), marker='$(a)$', color='black', markersize=15)
    if x == 'save_all':
        os.makedirs(f'{in_fd}/fig_6a_{dtdt}', exist_ok=True)
        plt.savefig(f'{in_fd}/fig_6a_{dtdt}/{AIA_304[-26:-5]}.png', bbox_inches='tight', dpi=200)
        plt.close()
    elif x == 'save_one':
        nama_file = 'fig_6a_{AIA_304[-26:-5]}.png'
        plt.savefig(f'{save_fd}/{nama_file}', bbox_inches='tight', dpi=200)
        print(f'1.1 Saving intenisty along line fig ("{save_fd}{nama_file}").')
        plt.close()
    plt.show()
    plt.close()
    
    return AIA_304, aia_map, intensity, data_date, intensity_coords, ht, ht_above, ht_below, angular_separation, angular_separation_above, angular_separation_below

## uncomment the codes below for resulting in images and intensity.npy and saving it. 
## don't forget to comment return (?) and plt.show() in function above, and the instance below
# [plot_6a(i, 'save_all') for i in range(len(data_list))]
# df.to_csv(f'{in_fd}/datetime_intensity_{dtdt}.csv')
# np.save(f'{in_fd}/intensity_{dtdt}', ht)
# np.save(f'{in_fd}/intensity_above_{dtdt}', ht_above)
# np.save(f'{in_fd}/intensity_below_{dtdt}', ht_below)

## an instance to load sdoaia304 cmap, only use aia_map variabel
AIA_304, aia_map, inte, dd, inco, ht, ht_ab, ht_b, ans, ans_ab, ans_be =  plot_6a(example_case, 'save_on')
# df_ang_sep = pd.DataFrame(ans, columns=['h_arcsec'])
# df_ang_sep_ab = pd.DataFrame(ans_ab, columns=['h_arcsec'])
# df_ang_sep_be = pd.DataFrame(ans_be, columns=['h_arcsec'])
# df_ang_sep.to_csv(f'{in_fd}/ang_sep_{dtdt}.csv', index=False)
# df_ang_sep_ab.to_csv(f'{in_fd}/ang_sep_ab_{dtdt}.csv', index=False)
# df_ang_sep_be.to_csv(f'{in_fd}/ang_sep_be_{dtdt}.csv', index=False)

sun_rad = aia_map.meta['RSUN_OBS'] #in arcsec
sun_rad_ref = aia_map.meta['RSUN_REF']/1000 #in km
one_arcsec_to_km = sun_rad_ref/sun_rad

intensity_mid = np.load(f"{in_fd}/intensity_{dtdt}.npy") #output from height_prom(i):
intensity_ab = np.load(f"{in_fd}/intensity_above_{dtdt}.npy") # for blue line
intensity_be = np.load(f"{in_fd}/intensity_below_{dtdt}.npy") # for green line
intensity = intensity_be
ref_line = 'below' # middle, above or below

dt_format = '%Y-%m-%dT%H:%M:%S.%f'
dt_intensity = pd.read_csv(f"{in_fd}/datetime_intensity_{dtdt}.csv", index_col='Unnamed: 0',parse_dates=['date'], date_format=dt_format)
dt_intensity['minute'] = dt_intensity['date'].apply(lambda i:i - dt_intensity['date'][0])
dt_intensity['minute'] = dt_intensity['minute'].apply(lambda i:i.seconds/60)
ang_sep_mid = pd.read_csv(f"{in_fd}/ang_sep_{dtdt}.csv")
ang_sep_ab = pd.read_csv(f"{in_fd}/ang_sep_ab_{dtdt}.csv")
ang_sep_be = pd.read_csv(f"{in_fd}/ang_sep_be_{dtdt}.csv")

ang_sep = ang_sep_be
ang_sep['h_km'] = ang_sep.h_arcsec*one_arcsec_to_km

start_time = dt_intensity.loc[0].date

def improve(inten):
    bin_factor = int(np.round(inten.shape[0] / 300))  # Calculate the bin factor
    image_binned = np.mean(inten.reshape(inten.shape[0] // bin_factor, bin_factor, inten.shape[1] // bin_factor, bin_factor), axis=(1, 3))
    # target_shape = (int(np.round(inten.shape[0] / bin_factor / 2)) * 2, int(np.round(inten.shape[1] / bin_factor / 2)) * 2)  # ~2" resolution
    # Assuming height_time_image is your input image
    # Step 1: Multiply each row by its height
    image_binned[:, 1] *= image_binned[:, 0]
    # Step 2: Calculate the median value
    median_value = np.median(image_binned)
    # Step 3: Threshold the image above 1.5x the median value
    threshold_value = 0.01 * median_value
    image_binned[image_binned < threshold_value] = np.nan
    return image_binned

def plot_param():
    fig, ax = plt.subplots()
    
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    date_format = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=80))
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    
    ax.axvline(dt_intensity.date[example_case],0,1, color = 'black', ls = '--')
    
    ax2 = ax.secondary_xaxis('top')
    ax2.xaxis.set_tick_params(labeltop=False)
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=80))

    ax3 = ax.secondary_yaxis('right')
    ax3.yaxis.set_tick_params(labelright=False)
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    
    # plt.axis('off')
    return fig, ax, ax2
    
imp = improve(intensity)

def plot_6b(x): #type x='save' to save figure, otherwise won't save anything
    '''
    This function is modified version of 1.2 height-time_plot.py
    
    '''
    xlim = [dt_intensity.date[0],dt_intensity.date[len(dt_intensity)-1]]
    x_lim = mdates.date2num(xlim)
    fig, ax, ax2 = plot_param()
    ax.imshow(imp, cmap=reversed_map, origin='lower', vmax=35, extent=[x_lim[0],x_lim[1],0,ang_sep.h_arcsec.max()], aspect='auto')
    ax.set_ylim(0,470)

    ax.set_ylabel('Distance along slice (arcsec)', fontsize=18)
    ax.set_xlabel('Start time = {}'.format(dt_intensity.date[0].strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)
    
    ax.text(dt_intensity.date[10],410,'(b)',fontsize=20)
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
    ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20))
    if x == 'save':
        name_file = f'fig_6b_wtline_{AIA_304[-26:-15]}.png'
        plt.savefig(f'{save_fd}/{name_file}',bbox_inches='tight', dpi=100)
        plt.close()
        print(f'1.2 Saving height-time fig without fitting line ("{save_fd}{name_file}").')
    plt.show()

plot_6b('sav')

arr = feature.canny(intensity, sigma=11)
## Detect contour all edges from array numpy with shape 182x307
coord = np.argwhere(arr==1)
df_coord = pd.DataFrame(coord, columns=['y', 'x'])
df_coord['y_arcsec'] = df_coord['y'].apply(lambda i:ang_sep.loc[i].h_arcsec)
df_coord['x_minute'] = df_coord['x'].apply(lambda i: dt_intensity.loc[i].minute)

df_coord = df_coord.sort_values(by=['x_minute'])
df_coord_edge = df_coord[df_coord.y_arcsec > (df_coord.y_arcsec[:5].max() - 1)]
df_coord_edge.drop_duplicates(subset=['y_arcsec'], inplace=True)
# df_coord_edge.to_csv('Results/canny_edge/edge_coord.csv')

start_time_edge = dt_intensity.loc[df_coord_edge.x.min()].date
coord_time = df_coord.x_minute.apply(lambda i:start_time + timedelta(minutes=i))
coord_edge_time = df_coord_edge.x_minute.apply(lambda i:start_time_edge + timedelta(minutes=i))

def plot_6c(x): #type x='save' to save figure, otherwise won't save anything
    '''
    This function is modified version of 1.3 canny.py
    
    '''
    fig, ax, ax2 = plot_param()
    ax.plot(coord_time, df_coord.y_arcsec, color='cyan', marker='.',linestyle='None', markersize=2)
    ax.plot(coord_edge_time, df_coord_edge.y_arcsec, color='red', marker='.', linestyle='None', markersize=2)
    ax.set_ylim(0,450)
    ax.set_ylabel('Height (arcsec)', fontsize=18)
    ax.set_xlabel('Start time = {}'.format(start_time.strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)

    ax.text(dt_intensity.date[7],400,'(c)',fontsize=20)
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
    ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20))
    if x == 'save':
        name_file = f'fig_6c_{AIA_304[-26:-15]}.png'
        plt.savefig(f'{save_fd}/{name_file}',bbox_inches='tight', dpi=100)
        plt.close()
        print(f'1.3 Saving edge detection ("{save_fd}{name_file}").')
    plt.show()

plot_6c('sav')

df_coord_edge['x_second'] = df_coord_edge.x_minute*60
df_coord_edge['y_Mm'] = df_coord_edge['y'].apply(lambda i:ang_sep.loc[i].h_km)/1000
df_coord_edge['y_km'] = df_coord_edge['y'].apply(lambda i:ang_sep.loc[i].h_km)
df_coord_edge.drop_duplicates(subset=['x'], inplace=True)

start_slice = 0
end_slice = df_coord_edge.shape[0] - 9
t_data = df_coord_edge.x_minute[start_slice:end_slice]
h_data_mm = df_coord_edge.y_Mm[start_slice:end_slice] #different data slicing resulting in different result

def model_h(t,c0,tau,c1,c2,t0):
    return c0 * np.exp((t-t0)/ tau) + c1 * (t-t0) + c2
def t_ons(tau,c1,c0,t0):
    return tau*np.log(c1*tau/c0)+t0

popt, pcov = curve_fit(model_h, t_data, h_data_mm, p0=[7,31,0.1,89,211])
c0_opt, tau_opt,c1_opt,c2_opt,t0_opt = popt
t_model = np.linspace(min(t_data), max(t_data), int(max(t_data)-min(t_data)))
h_model_mm = model_h(t_model,c0_opt,tau_opt,c1_opt,c2_opt, t0_opt)
dft_model = pd.Series(t_model)
model_t = dft_model.apply(lambda i:start_time_edge + timedelta(minutes=i))
ti = t_data.apply(lambda i:start_time_edge + timedelta(minutes=i))

t_onset = t_ons(tau_opt, c1_opt, c0_opt, t0_opt)
print(f't_onset:{t_onset}')
onset_t = t_model[t_model >= t_onset][0]
# start_time = dt_intensity[dt_intensity.minute >= t_model[0]].date.min()
onset_time = start_time_edge + timedelta(minutes=onset_t)
onset_height = df_coord_edge[df_coord_edge.x_minute >= onset_t].y_Mm.values[0]
onset_height_arcsec = df_coord_edge[df_coord_edge.x_minute >= onset_t].y_arcsec.values[0]
onset_h = h_model_mm[t_model >= onset_t][0]
onset_h_arcsec = df_coord_edge[df_coord_edge.y_Mm >= onset_h].y_arcsec.values[0]

def plot_6d(x):
    '''
    This function is modified version of 1.4 fitting_height.py
    
    '''
    fig, ax, ax2 = plot_param()
    ax.plot(model_t,h_model_mm, color='k')
    ax.set_ylim(0)
    ax.set_ylabel('Height (Mm)', fontsize=18)
    ax.set_xlabel('Start time = {}'.format(start_time.strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)

    ax.text(dt_intensity.date[5],285,'(d)',fontsize=20)

    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    
    if x == 'save':
        name_file = f'fig_6d_{AIA_304[-26:-15]}.png'
        plt.savefig(f'{save_fd}/{name_file}',bbox_inches='tight', dpi=100)
        plt.close()
        print(f'1.4 Saving fitting height ("{save_fd}{name_file}").')
    plt.show()

plot_6d('sav')

t,c0,tau,c1,c2,t0 = smp.symbols('t c0 tau c1 c2 t0', real=True)
eq_h = c0 * smp.exp((t-t0)/tau) + c1*(t-t0) + c2
dhdt = smp.diff(eq_h,t)
# print(f'v(0) : {dhdt.subs([(t,8),(c0,c0_opt),(tau,tau_opt),(c1, c1_opt),(t0,t0_opt)]).evalf()}')
dhdt_f = smp.lambdify((t,c0,tau,c1,t0), dhdt)
hv = dhdt_f(t_model,c0=c0_opt,tau=tau_opt,c1=c1_opt,t0=t0_opt)
hv = hv*1000/60 ## in km/s

def plot_6e(x):
    '''
    This function is modified version of 1.4.1 fitting_velocity.py
    
    '''
    fig, ax, ax2 = plot_param()
    ax.plot(model_t,hv, color='k')
    ax.set_ylim(0,120)
    ax.set_ylabel('Velocity (km $s^-1$)', fontsize=18)
    ax.set_xlabel('Start time = {}'.format(start_time.strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)
    ax.text(dt_intensity.date[5],100,'(e)',fontsize=20)

    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    
    if x == 'save':
        name_file = f'fig_6e_{AIA_304[-26:-15]}.png'
        plt.savefig(f'{save_fd}/{name_file}',bbox_inches='tight', dpi=100)
        plt.close()
        print(f'1.5 Saving fitting velocity ("{save_fd}{name_file}").')
    plt.show()

plot_6e('sav')

### Calculate Acceleration
d2hdt2 = smp.diff(eq_h,t,2)
d2hdt2_f = smp.lambdify((t,c0,tau,c1,t0), d2hdt2)
ha = d2hdt2_f(t_model,c0=c0_opt,tau=tau_opt,c1=c1_opt,t0=t0_opt)
ha = ha *1000/60/60
a_o = ha[model_t[model_t >= onset_time].index[0]] 
## because of ha shape is equal to model_t not df_coord_edge
## we using model_t instead of df_coord_edge

def plot_6f(x):
    '''
    This function is modified version of 1.4.2 fitting_acceleration.py
    
    '''
    fig, ax, ax2 = plot_param()
    ax.plot(model_t,ha, color='k')
    ax.set_ylim(0,0.05)
    ax.set_ylabel('Accel. (km $s^-2$)', fontsize=18)
    ax.set_xlabel('Start time = {}'.format(start_time.strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)
    
    ax.text(dt_intensity.date[5],0.042,'(f)',fontsize=20)
    
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.002))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    
    if x == 'save':
        name_file = f'fig_6f_{AIA_304[-26:-15]}.png'
        plt.savefig(f'{save_fd}/{name_file}',bbox_inches='tight', dpi=100)
        plt.close()
        print(f'1.6 Saving fitting acceleration ("{save_fd}{name_file}").')
    plt.show()

plot_6f('sav')

def plot_6b_comp(x):
    '''
    This function is modified version of 1.6 ht_plot_comp.py
    
    '''
    fig, ax, ax2 = plot_param()
    h_data_arcsec = df_coord_edge.y_arcsec[start_slice:end_slice]
    popt, pcov = curve_fit(model_h, t_data, h_data_arcsec, p0=[7,31,0.1,89,211]) #p0=[355,128,0,336]
    c0_opt, tau_opt,c1_opt,c2_opt,t0_opt = popt
    h_model_arcsec = model_h(t_model,c0_opt,tau_opt,c1_opt,c2_opt, t0_opt)
    dft_model = pd.Series(t_model)
    model_t = dft_model.apply(lambda i:start_time + timedelta(minutes=i))

    t_onset = t_ons(tau_opt, c1_opt, c0_opt, t0_opt)
    onset_t = t_model[t_model >= t_onset][0]
    onset_time = start_time_edge + timedelta(minutes=onset_t)
    onset_height = df_coord_edge[df_coord_edge.x_minute >= onset_t].y_Mm.values[0]
    onset_height_arcsec = df_coord_edge[df_coord_edge.x_minute >= onset_t].y_arcsec.values[0]
    onset_h_arcsec = h_model_arcsec[t_model >= onset_t][0]
    onset_h_Mm = df_coord_edge[df_coord_edge.y_arcsec>= onset_h_arcsec].y_Mm.values[0]
    
    xlim = [dt_intensity.date[0],dt_intensity.date[len(dt_intensity)-1]]
    # end_time = start_time + timedelta(minutes=t_data.values[-1])
    # xlim = [start_time,end_time]
    x_lim = mdates.date2num(xlim)
    ax.imshow(imp, cmap=reversed_map, origin='lower', vmax=35, extent=[x_lim[0],x_lim[1],0,ang_sep.h_arcsec.max()], aspect='auto')
    # ax.plot(coord_edge_time[3:], df_coord_edge.y_arcsec[3:], color='k')
    ax.plot(model_t[100:], h_model_arcsec[100:], color='k')
    ax.set_ylim(0,470)
    ax.set_ylabel('Distance along slice (arcsec)', fontsize=18)
    ax.set_xlabel('Start time = {}'.format(start_time.strftime('%Y/%m/%d %H:%M:%S')), fontsize=18)

    ax.text(dt_intensity.date[10],410,'(b)',fontsize=20)
    ax.text(onset_time-timedelta(minutes=45),onset_h_arcsec+13,f'{round(onset_h)} Mm',fontsize=13)
    ax.plot(onset_time,onset_h_arcsec, 'k*', ms=10)

    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20)) 
    ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=20))
    if x == 'save':
        name_file = f'fig_6b_comp_{AIA_304[-26:-15]}.png'
        plt.savefig(f'{save_fd}/{name_file}',bbox_inches='tight', dpi=100)
        plt.close()
        print(f'1.2 Saving height-time fig with fitting line ("{save_fd}{name_file}").')
    plt.show()

plot_6b_comp('sav')

print(f'var c0: {c0_opt}')
print(f'var tau: {tau_opt}')
print(f'var c1: {c1_opt}')
print(f'var c2: {c2_opt}')
print(f'var t0: {t0_opt}')
print('onset time: {} UT'.format(onset_time.strftime('%Y/%m/%d %H:%M')))
print(f'onset height: {onset_h} Mm')
print(f'onset height: {onset_h_arcsec} arcsec')
print(f'Initial slow rise velocity: {hv[0]} km s-1')
print(f'Maximum velocity in the AIA FOV: {hv[-1]} km s-1')
print(f'Acceleration at the onset point: {a_o*1000} m s^-2')
print(f'Final acceleration: {ha[-1]*1000} m s^-2')
param = np.array([[c0_opt,tau_opt, c1_opt, c2_opt,t0_opt, onset_time, onset_t, onset_h, onset_h_arcsec, hv[0], hv[-1], a_o*1000, ha[-1]*1000, ref_line]])
df_param = pd.DataFrame(param, columns=['c0', 'tau', 'c1', 'c2', 't0', 'onset_time', 'onset_t_minute', 'onset_h_mm', 'onset_h_arcsec', 'init_slow_v', 'max_v', 'onset_acc', 'end_acc', 'ref_line'], index=None)
df_param.to_csv(f'{save_fd}/parameter.csv', mode='a', index=False, header=False)
