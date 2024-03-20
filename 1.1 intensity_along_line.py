import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.map
import numpy as np
import os
import pandas as pd
import matplotlib

# matplotlib.use('Agg')
# AIA_304 = r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Data\AIA304\2012\AIA20120312_0104_0304.fits"
# AIA_304 = r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Data\AIA304\2012\AIA20120310_0900_0304.fits"
# AIA_304 = r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Data\AIA304\2012\AIA20120311_2208_0304.fits"
# AIA_304 = r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Data\AIA304\2012\AIA20120311_2320_0304.fits"
data_list = os.listdir('Data/AIA304/2012')
df = pd.DataFrame()
ht = np.zeros([307,len(data_list)]) #307 is array black line.
ht_above = np.zeros([315,len(data_list)]) #315 is array of blue line
ht_below = np.zeros([299,len(data_list)]) #299 is array of green line
## number of array should determined manually.
## you cannot assume all lines have the same array. I will raise an error.
# i = 0
def height_prom(i):
    AIA_304 = f'Data/AIA304/2012/{data_list[i]}'
    aia_map = sunpy.map.Map(AIA_304)
    # sun_rad = aia_map.meta['RSUN_OBS'] #in arcsec
    # sun_rad_ref = aia_map.meta['RSUN_REF']/1000 #in km
    # pixel_size = 0.6 #in arcsec from Lemen et al (2012)
    # one_arcsec_to_km = sun_rad_ref/sun_rad
    # one_pixel_to_km = pixel_size * one_arcsec_to_km #for further need
    
    xlims_world = [-900, -380]*u.arcsec 
    ylims_world = [-1200, -630]*u.arcsec
    world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world, frame=aia_map.coordinate_frame)
    pixel_coords_x, pixel_coords_y = aia_map.wcs.world_to_pixel(world_coords)
    
    x_start, y_start = -500, -840 #coordinate for the initial line at the solar surface
    x_end, y_end = -875, -1200 #coordinate for the end line 
    xe_above, ye_above = -895, -1200
    xe_below, ye_below = -855, -1200
    line_coords = SkyCoord([x_start, x_end], [y_start, y_end], unit=(u.arcsec, u.arcsec),
                           frame=aia_map.coordinate_frame)
    intensity_coords = sunpy.map.pixelate_coord_path(aia_map, line_coords)
    intensity = sunpy.map.sample_at_coords(aia_map, intensity_coords)
    
    line_coords_above = SkyCoord([x_start, xe_above], [y_start, ye_above], unit=(u.arcsec, u.arcsec),
                           frame=aia_map.coordinate_frame)
    intensity_coords_above = sunpy.map.pixelate_coord_path(aia_map, line_coords_above)
    intensity_above = sunpy.map.sample_at_coords(aia_map, intensity_coords_above)
    
    line_coords_below = SkyCoord([x_start, xe_below], [y_start, ye_below], unit=(u.arcsec, u.arcsec),
                           frame=aia_map.coordinate_frame)
    intensity_coords_below = sunpy.map.pixelate_coord_path(aia_map, line_coords_below)
    intensity_below = sunpy.map.sample_at_coords(aia_map, intensity_coords_below)
    
    data_date = aia_map.date.value
    df.loc[i, 'date'] = data_date
    
    ht[:,i] = intensity
    ht_above[:,i] = intensity_above
    ht_below[:,i] = intensity_below
    '''
    orig_map=matplotlib.colormaps.get_cmap('sdoaia304')
    reversed_map = orig_map.reversed()
    fig = plt.figure()
    ax1 = fig.add_subplot(projection=aia_map)
    aia_map.plot(axes=ax1, cmap=reversed_map, clip_interval=(1, 99.9)*u.percent)
    ax1.set_xlim(pixel_coords_x)
    ax1.set_ylim(pixel_coords_y)
    ax1.plot_coord(intensity_coords, lw=1, color='black')
    ax1.plot_coord(intensity_coords_above, lw=1, color='blue')
    ax1.plot_coord(intensity_coords_below, lw=1, color='green')
    plt.tight_layout()
    plt.savefig(f'Results/intensity_along_line/fig_6a/{AIA_304[-26:-5]}.png', dpi=200)
    # plt.show()
    plt.close()
    '''
    return aia_map, intensity, data_date, intensity_coords, ht, ht_above, ht_below
[height_prom(i) for i in range(len(data_list))]
# aia_map, intensity, data_date, intensity_coords, ht, ht_above, ht_below =  height_prom(160)
# df.to_csv('Results/intensity_along_line/date.csv, index=False)
# np.save('Results/intensity_along_line/intensity', ht) #save intensity to file
# np.save('Results/intensity_along_line/intensity_above', ht_above)
# np.save('Results/intensity_along_line/intensity_below', ht_below)
