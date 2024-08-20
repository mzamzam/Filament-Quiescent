'''
modified by MZN
original by TD, JM
'''
import numpy as np
import pandas as pd
import os
import sunpy.map
from scipy.optimize import curve_fit
from datetime import timedelta

dtdt = '20120312'
ins = 'AIA304'
data_list = sorted(os.listdir(f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}'))
AIA_304 = f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}/{data_list[50]}'
aia_map = sunpy.map.Map(AIA_304)
sun_rad = aia_map.meta['RSUN_OBS']   ## in arcsec
sun_rad_ref = aia_map.meta['RSUN_REF']/1000   ## in km
one_arcsec_to_km = sun_rad_ref/sun_rad
# Define the model function
def eq_h(t,c0,tau,c1,c2,t0):
    ## this is equation (1) from Su et al (2015)
    return c0 * np.exp((t-t0)/ tau) + c1 * (t-t0) + c2
def t_ons(tau,c1,c0,t0):
    ## this is equation (2) from Su et al (2015)
    return tau*np.log(c1*tau/c0)+t0
ang_sep = pd.read_csv(f"Results/intensity_along_line/ang_sep_{dtdt}.csv")
ang_sep['h_km'] = ang_sep.h_arcsec*round(one_arcsec_to_km)
dt_intensity = pd.read_csv(f"Results/intensity_along_line/datetime_intensity_{dtdt}.csv",parse_dates=['date'], date_format='%Y-%m-%dT%H:%M:%S.%f')
dt_intensity['minute'] = dt_intensity['date'].apply(lambda i:i - dt_intensity['date'][0])
dt_intensity['minute'] = dt_intensity['minute'].apply(lambda i:i.seconds/60)
start_time = dt_intensity.loc[1].date
edge_coord = pd.read_csv(f"Results/canny/edge_coord_{dtdt}.csv", index_col='Unnamed: 0')
edge_coord['x_second'] = edge_coord.x_minute*60
edge_coord['y_Mm'] = edge_coord['y'].apply(lambda i:ang_sep.loc[i].h_km)/1000
edge_coord.drop_duplicates(subset=['x'], inplace=True)
save_var = pd.DataFrame()
## Changing data range, fix init param to get t_onsest (only use 1 for loop)
## another way is changing init param, fix data range. However, it will require more than one for loop
for i in range (edge_coord.shape[0]-20,edge_coord.shape[0],1):
    start_slice = 0
    end_slice = i  ## or can be used: edge_coord.shape[0]
    # Load the dataset
    t_minute = edge_coord['x_minute'][start_slice:end_slice]
    h_Mm = edge_coord['y_Mm'][start_slice:end_slice]
    # HEURISTIC INITIALIZATION USING DATA-DRIVEN FOR GUESSING THE INITIAL PARAMETERS
    c0 = h_Mm.iloc[0]
    t0 = t_minute.min()
    tau = (t_minute.max()) - (t_minute.min()) / 10
    c1 = 0.1#(h_Mm.iloc[-1] - h_Mm.iloc[0]) / (t_minute.iloc[-1] - t_minute.iloc[0])
    c2 = 0
    try:
        popt, pcov = curve_fit(eq_h, t_minute, h_Mm, p0=[c0, tau, c1, c2, t0])
        c0_opt, tau_opt,c1_opt,c2_opt,t0_opt = popt
    except RuntimeError:
        continue
    t_minute_model = np.linspace(min(t_minute), max(t_minute), int(max(t_minute)-min(t_minute)))
    h_Mm_model = eq_h(t_minute_model,c0_opt,tau_opt,c1_opt,c2_opt, t0_opt)
    dft_model = pd.Series(t_minute_model)
    t_minute_onset = t_ons(tau_opt, c1_opt, c0_opt, t0_opt)
    if t_minute_onset != np.nan:
        try:
            t_minute_onset_model = t_minute_model[t_minute_model >= t_minute_onset][0]
            t_onset_datetime = start_time + timedelta(minutes=t_minute_onset_model)
            h_onset_Mm_data = edge_coord[edge_coord.x_minute >= t_minute_onset_model].y_Mm.values[0]
            h_onset_arcsec_datax = edge_coord[edge_coord.x_minute >= t_minute_onset_model].y_arcsec.values[0]
            h_onset_Mm_model = h_Mm_model[np.argwhere(t_minute_model == t_minute_onset_model)[0][0]]
            print(f'optimum parameter (c0, tau, c1, c2, t0): {round(c0_opt, 3), round(tau_opt, 3), round(c1_opt, 3), round(c2_opt, 3), round(t0_opt, 3)}')
            print('onset time: {} UT'.format(t_onset_datetime.strftime('%Y/%m/%d %H:%M')))
            print(f'onset height: {h_onset_Mm_model} Mm')
            print(f'end slice for opt param:{i}; ori shape: {edge_coord.shape[0]}')
            save_var.loc[i, 'data_shape'] = edge_coord.shape[0]
            save_var.loc[i, 'end_slice'] = i
            save_var.loc[i, 'c0_opt'] = c0_opt
            save_var.loc[i, 'tau_opt'] = tau_opt
            save_var.loc[i, 'c1_opt'] = c1_opt
            save_var.loc[i, 'c2_opt'] = c2_opt
            save_var.loc[i, 't0_opt'] = t0_opt
            save_var.loc[i, 'onset_time'] = t_onset_datetime
            save_var.loc[i, 'onset_h_Mm'] = h_onset_Mm_model
        except IndexError:
            save_var.loc[i, 'data_shape'] = edge_coord.shape[0]
            save_var.loc[i, 'end_slice'] = i
            save_var.loc[i, 'c0_opt'] = c0_opt
            save_var.loc[i, 'tau_opt'] = tau_opt
            save_var.loc[i, 'c1_opt'] = c1_opt
            save_var.loc[i, 'c2_opt'] = c2_opt
            save_var.loc[i, 't0_opt'] = t0_opt
            save_var.loc[i, 'onset_time'] = t_minute_onset
            save_var.loc[i, 'onset_h_Mm'] = np.nan
            continue
print(save_var)
os.makedirs('Results/fitting_height', exist_ok=True)
save_var.to_csv('Results/fitting_height/find_initparam.csv', index=False)