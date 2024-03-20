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
ht = np.zeros([307,len(data_list)]) #307 is array number of selected line.
# i = 0
def height_prom(i):
    AIA_304 = f'Data/AIA304/2012/{data_list[i]}'
    aia_map = sunpy.map.Map(AIA_304)
    sun_rad = aia_map.meta['RSUN_OBS'] #in arcsec
    sun_rad_ref = aia_map.meta['RSUN_REF']/1000 #in km
    # pixel_size = 0.6 #in arcsec from Lemen et al (2012)
    one_arcsec_to_km = sun_rad_ref/sun_rad
    # one_pixel_to_km = pixel_size * one_arcsec_to_km #for further need
    
    xlims_world = [-900, -380]*u.arcsec 
    ylims_world = [-1200, -630]*u.arcsec
    world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world, frame=aia_map.coordinate_frame)
    pixel_coords_x, pixel_coords_y = aia_map.wcs.world_to_pixel(world_coords)
    
    x_start, y_start = -500, -840 #coordinate for the initial line at the solar surface
    x_end, y_end = -875, -1200 #coordinate for the end line 
    line_coords = SkyCoord([x_start, x_end], [y_start, y_end], unit=(u.arcsec, u.arcsec),
                           frame=aia_map.coordinate_frame)
    intensity_coords = sunpy.map.pixelate_coord_path(aia_map, line_coords)
    intensity = sunpy.map.sample_at_coords(aia_map, intensity_coords)
    angular_separation = intensity_coords.separation(intensity_coords[0]).to(u.arcsec)
    threshold = 3 # from the histogram of intensity  
    mask = np.where(intensity > threshold)        
    height = angular_separation[np.max(mask)] #in arcsec
    height_km = height.value * one_arcsec_to_km #in km
    height_coords = intensity_coords[np.max(mask)]
    data_date = aia_map.date.value
    df.loc[i, 'height'] = height.value
    df.loc[i, 'height_km'] = height_km
    df.loc[i, 'date'] = data_date

    ht[:,i] = intensity
    '''
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121, projection=aia_map)
    aia_map.plot(axes=ax1, clip_interval=(1, 99.9)*u.percent)
    ax1.set_xlim(pixel_coords_x)
    ax1.set_ylim(pixel_coords_y)
    ax1.plot_coord(intensity_coords)
    ax1.plot_coord(height_coords, marker=".", color="green", label="height")
    
    ax2 = fig.add_subplot(122)
    ax2.plot(angular_separation, intensity, '.')
    ax2.set_xlabel("Angular distance along slit [arcsec]")
    ax2.set_ylabel(f"Intensity [{aia_map.unit}]")
    plt.savefig(f'Results/{AIA_304[-26:-5]}.png', dpi=300)
    plt.close(fig)
    '''
    return aia_map, intensity, angular_separation, mask, height, height_coords, data_date, intensity_coords, ht
[height_prom(i) for i in range(len(data_list))]
aia_map, intensity, angular_separation, mask, height, height_coords, data_date, intensity_coords, ht =  height_prom(160)
# np.save('Results/intensity_along_line/intensity', ht) #save intensity to file

'''
# Saving histogram
plt.show()
plt.hist(intensity, label=AIA_304[-26:-5]) #make histogram of intensity data.
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(f'Results/hist2_{AIA_304[-26:-5]}.png', dpi=300)
'''