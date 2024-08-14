import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.map
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')    ## don't display plot box
dtdt = '20120312'        ## YYYYMMDD, usually day of the eruption
ins = 'AIA304'           ## instrument
data_list = os.listdir(f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}')  ## using 'f' because there is a variable on string
df = pd.DataFrame()      ## empty dataframe array (only for several columns)
ht = np.zeros([307,217]) ## empty numpy array (for huge array) with size 307x230 for main line; 307 is length of main lain (guess first), 230 is number of data in data_list
def height_prom(i):
    AIA_304 = f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}/{data_list[i]}'  ## i-th data
    aia_map = sunpy.map.Map(AIA_304)    ## read fits data
    xlims_world = [-900, -380]*u.arcsec   ## x-coordinte for crop size; use 1.1.0 to get the numbers
    ylims_world = [-1200, -630]*u.arcsec  ## y-coordinate for crop size; use 1.1.0 to get the numbers
    world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world, frame=aia_map.coordinate_frame)
    pixel_coords_x, pixel_coords_y = aia_map.wcs.world_to_pixel(world_coords)   ## change crop area coordinate so that it can be drawn in AIA coordinate
    x_start, y_start = -500, -840         ## coordinate of the initial main line on the solar surface
    x_end, y_end = -875, -1200            ## coordinate of the end main line.
    line_coords = SkyCoord([x_start, x_end], [y_start, y_end], unit=(u.arcsec, u.arcsec),
                           frame=aia_map.coordinate_frame)   ## change coordinate of main line so that it can be drawn in AIA coordinate
    intensity_coords = sunpy.map.pixelate_coord_path(aia_map, line_coords)   ## coordinate of the main line on AIA frame
    intensity = sunpy.map.sample_at_coords(aia_map, intensity_coords)        ## intensity passed by main line
    angular_separation = intensity_coords.separation(intensity_coords[0]).to(u.arcsec)   ## distance of points in the main line
    data_date = aia_map.date.value   ## date of i-th data
    df.loc[i, 'date'] = data_date    ## save date of i-th data into empty dataframe array
    ht[:,i] = intensity              ## save i-th intensity into empty numpy array
    orig_map=matplotlib.colormaps.get_cmap('sdoaia304')    ## call SDO/AIA 304 color map
    reversed_map = orig_map.reversed()                     ## color reverse of SDO/AIA 304 color map
    fig = plt.figure()
    ax1 = fig.add_subplot(projection=aia_map)
    aia_map.plot(axes=ax1, cmap=reversed_map, clip_interval=(1, 99.9)*u.percent)
    ax1.set_xlim(pixel_coords_x)
    ax1.set_ylim(pixel_coords_y)
    ax1.plot_coord(intensity_coords, lw=1, color='black')
    ax1.plot_coord(SkyCoord(-845*u.arcsec, -700*u.arcsec, frame=aia_map.coordinate_frame), marker='$(a)$', color='black', markersize=15)   ## add anotate (a) on plot
    plt.tight_layout()
    os.makedirs(f'Results/intensity_along_line/fig_6a_{dtdt}_{ins}', exist_ok=True)   ## make new directory if not present
    plt.savefig(f'Results/intensity_along_line/fig_6a_{dtdt}_{ins}/{AIA_304[-26:-5]}.png', dpi=200)
    print(f'Saving image #{i} from #{len(data_list)}')
    plt.close()
    return aia_map, intensity, data_date, intensity_coords, ht, angular_separation
[height_prom(i) for i in range(len(data_list))]   ## for loop pythonic way
df.to_csv(f'Results/intensity_along_line/datetime_intensity_{dtdt}.csv', index=False)   ## save data date array to csv file in Results/intensity_along_line folder
np.save(f'Results/intensity_along_line/intensity_{dtdt}', ht)                                 ## save intensity data to numpy array in Results/intensity_along_line folder
aia_map, intensity, data_date, intensity_coords, ht, ans =  height_prom(10)   ## sorted same as return in line-46, 10 is choosen randomly as long as do not exceed number of data
df_ang_sep = pd.DataFrame(ans, columns=['h_arcsec'])   ## save distance of points into dataframe array
df_ang_sep.to_csv(f'Results/intensity_along_line/ang_sep_{dtdt}.csv', index=False)     ## save into csv file



