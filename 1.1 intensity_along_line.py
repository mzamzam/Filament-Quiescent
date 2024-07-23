import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.map
import numpy as np
import os
import pandas as pd
import matplotlib

matplotlib.use('Agg')
# AIA_304 = r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Data\AIA304\2012\AIA20120312_0104_0304.fits"
# AIA_304 = r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Data\AIA304\2012\AIA20120310_0900_0304.fits"
# AIA_304 = r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Data\AIA304\2012\AIA20120311_2208_0304.fits"
# AIA_304 = r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Data\AIA304\2012\AIA20120311_2320_0304.fits"
dtdt = '20230420'
ins = 'AIA304'
data_list = os.listdir(f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}')
# data_list = os.listdir('Data/AIA304/2012')
df = pd.DataFrame()
ht = np.zeros([201,290]) #307 is array black line.
# ht_above = np.zeros([577,128]) #315 is array of blue line
# ht_below = np.zeros([628,128]) #299 is array of green line
## number of array should determined manually.
## you cannot assume all lines have the same array. It will raise an error.
# i = 0
def height_prom(i):
    AIA_304 = f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}/{data_list[i]}'
    # AIA_304 = f'Data/AIA304/2012/{data_list[i]}'
    aia_map = sunpy.map.Map(AIA_304)
    # sun_rad = aia_map.meta['RSUN_OBS'] #in arcsec
    # sun_rad_ref = aia_map.meta['RSUN_REF']/1000 #in km
    # pixel_size = 0.6 #in arcsec from Lemen et al (2012)
    # one_arcsec_to_km = sun_rad_ref/sun_rad
    # one_pixel_to_km = pixel_size * one_arcsec_to_km #for further need
    
    xlims_world = [200, 550]*u.arcsec 
    ylims_world = [750, 1200]*u.arcsec
    world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world, frame=aia_map.coordinate_frame)
    pixel_coords_x, pixel_coords_y = aia_map.wcs.world_to_pixel(world_coords)
    
    x_start, y_start = 340, 905 #coordinate for the initial line at the solar surface
    x_end, y_end = 525, 1200 #coordinate for the end line 
    # xe_above, ye_above = -2300, 3900 #blue
    # xe_below, ye_below = -2300, 3900 #green
    
    # line_coords = SkyCoord([x_start, x_end], [y_start, y_end], unit=(u.arcsec, u.arcsec),
    #                        frame=aia_map.coordinate_frame)
    # intensity_coords = sunpy.map.pixelate_coord_path(aia_map, line_coords)
    # intensity = sunpy.map.sample_at_coords(aia_map, intensity_coords)
    # angular_separation = intensity_coords.separation(intensity_coords[0]).to(u.arcsec)
    # line_coords_above = SkyCoord([x_start, xe_above], [y_start, ye_above], unit=(u.arcsec, u.arcsec),
    #                         frame=aia_map.coordinate_frame)
    # intensity_coords_above = sunpy.map.pixelate_coord_path(aia_map, line_coords_above)
    # intensity_above = sunpy.map.sample_at_coords(aia_map, intensity_coords_above)
    
    # line_coords_below = SkyCoord([x_start, xe_below], [y_start, ye_below], unit=(u.arcsec, u.arcsec),
    #                         frame=aia_map.coordinate_frame)
    # intensity_coords_below = sunpy.map.pixelate_coord_path(aia_map, line_coords_below)
    # intensity_below = sunpy.map.sample_at_coords(aia_map, intensity_coords_below)
    
    line_coords = SkyCoord([x_start, x_end], [y_start, y_end], unit=(u.arcsec, u.arcsec),
                           frame=aia_map.coordinate_frame)
    intensity_coords = sunpy.map.pixelate_coord_path(aia_map, line_coords)
    intensity = sunpy.map.sample_at_coords(aia_map, intensity_coords)
    angular_separation = intensity_coords.separation(intensity_coords[0]).to(u.arcsec)
    
    # line_coords_above = SkyCoord([x_start, xe_above], [y_start, ye_above], unit=(u.arcsec, u.arcsec),
    #                         frame=aia_map.coordinate_frame)
    # intensity_coords_above = sunpy.map.pixelate_coord_path(aia_map, line_coords_above)
    # intensity_above = sunpy.map.sample_at_coords(aia_map, intensity_coords_above)
    # angular_separation_above = intensity_coords_above.separation(intensity_coords_above[0]).to(u.arcsec)
    
    # line_coords_below = SkyCoord([x_start, xe_below], [y_start, ye_below], unit=(u.arcsec, u.arcsec),
    #                         frame=aia_map.coordinate_frame)
    # intensity_coords_below = sunpy.map.pixelate_coord_path(aia_map, line_coords_below)
    # intensity_below = sunpy.map.sample_at_coords(aia_map, intensity_coords_below)
    # angular_separation_below = intensity_coords_below.separation(intensity_coords_below[0]).to(u.arcsec)
    
    data_date = aia_map.date.value
    df.loc[i, 'date'] = data_date
    
    ht[:,i] = intensity
    # ht_above[:,i] = intensity_above
    # ht_below[:,i] = intensity_below
    
    orig_map=matplotlib.colormaps.get_cmap('sdoaia304')
    reversed_map = orig_map.reversed()
    fig = plt.figure()
    ax1 = fig.add_subplot(projection=aia_map)
    aia_map.plot(axes=ax1, cmap=reversed_map, clip_interval=(1, 99.9)*u.percent)
    ax1.set_xlim(pixel_coords_x)
    ax1.set_ylim(pixel_coords_y)
    ax1.plot_coord(intensity_coords, lw=1, color='black')
    # ax1.plot_coord(intensity_coords_above, lw=1, color='blue')
    # ax1.plot_coord(intensity_coords_below, lw=1, color='green')
    # ax1.plot_coord(SkyCoord(-845*u.arcsec, -700*u.arcsec, frame=aia_map.coordinate_frame), marker='$(a)$', color='black', markersize=15)
    plt.tight_layout()
    os.makedirs(f'Results/intensity_along_line/fig_6a_{dtdt}_{ins}', exist_ok=True)
    # plt.savefig(f'Results/intensity_along_line/fig_6a_{dtdt}_{ins}/{AIA_304[-26:-5]}.png', dpi=200)
    print(f'Saving image #{i} from #{len(data_list)}')
    # plt.show()
    plt.close()
    
    # return aia_map, intensity, data_date, intensity_coords, ht, ht_above, ht_below, angular_separation, angular_separation_above, angular_separation_below
    return aia_map, intensity, data_date, intensity_coords, ht, angular_separation
[height_prom(i) for i in range(len(data_list))]
# [height_prom(i) for i in range(len(data_list)-10,len(data_list),1)]
# df.to_csv(f'Results/intensity_along_line/datetime_intensity_{dtdt}.csv', index=False)
# np.save(f'Results/intensity_along_line/intensity_above_{dtdt}', ht_above)
# np.save(f'Results/intensity_along_line/intensity_below_{dtdt}', ht_below)
# np.save(f'Results/intensity_along_line/intensity_{dtdt}', ht) #save intensity to file


# aia_map, intensity, data_date, intensity_coords, ht, ht_above, ht_below, ans, ans_ab, ans_be=  height_prom(10)
# df_ang_sep = pd.DataFrame(ans, columns=['h_arcsec'])
# df_ang_sep_ab = pd.DataFrame(ans_ab, columns=['h_arcsec'])
# df_ang_sep_be = pd.DataFrame(ans_be, columns=['h_arcsec'])
# df_ang_sep.to_csv(f'Results/intensity_along_line/ang_sep_{dtdt}.csv', index=False)
# df_ang_sep_ab.to_csv(f'Results/intensity_along_line/ang_sep_ab_{dtdt}.csv', index=False)
# df_ang_sep_be.to_csv(f'Results/intensity_along_line/ang_sep_be_{dtdt}.csv', index=False)



