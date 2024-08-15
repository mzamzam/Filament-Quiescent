"""
This is not main code.

Trials and errors for finding the right crop size and coordinate of our region of interest can be done here.
Also, to find the right coordinate for the error lines (blue and green line).

If you wish to add error lines, simply uncomment all the commented-lines and delete ''' on the line-49 and 61.

"""
import sunpy.map
import os
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import math

dtdt = '20120312'
ins = 'AIA304'
data_list = sorted(os.listdir(f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}'))
print(f'Number of data: {len(data_list)}')
print(f'First data: {data_list[0]}')
print(f'Last data: {data_list[-1]}')
i = 100
print(f'Current data: {data_list[i]}')
AIA_304 = f'Data/{ins}/{dtdt[:4]}/{dtdt[4:6]}/{data_list[i]}'
aia_map = sunpy.map.Map(AIA_304)
coords = SkyCoord(Tx=(-900, -380) * u.arcsec,Ty=(-1200, -630) * u.arcsec,frame=aia_map.coordinate_frame)
x_start, y_start = -500, -840   ## coordinate for the initial line at the solar surface
x_end, y_end = -875, -1200       ##coordinate for the end line
#xe_above, ye_above = -2300, 3900 #blue
#xe_below, ye_below = -2300, 3900 #green
#xs_above, ys_above = xs_below, ys_below = x_start, y_start
def find_angle(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calculate the slopes of the two lines
    m1 = (y2 - y1) / (x2 - x1)
    m2 = (y4 - y3) / (x4 - x3)
    # Calculate the angle between the two lines
    angle = math.atan((m2 - m1) / (1 + m1 * m2))
    # Convert the angle from radians to degrees
    angle = math.degrees(angle)
    return angle
#angle_above = find_angle(x_start, y_start, x_end, y_end, xs_above, ys_above, xe_above, ye_above)
#angle_below = find_angle(x_start, y_start, x_end, y_end, xs_above, ys_above, xe_below, ye_below)
#print(f"The angle between ref-above line is {angle_above} (blue) degrees.")
#print(f"The angle between ref-below line is {angle_below} (green) degrees.")
line_coords = SkyCoord([x_start, x_end], [y_start, y_end], unit=(u.arcsec, u.arcsec),
                       frame=aia_map.coordinate_frame)
intensity_coords = sunpy.map.pixelate_coord_path(aia_map, line_coords)
intensity = sunpy.map.sample_at_coords(aia_map, intensity_coords)
angular_separation = intensity_coords.separation(intensity_coords[0]).to(u.arcsec)
'''
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
'''
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection=aia_map)
aia_map.plot(axes=ax, clip_interval=(1, 99.99)*u.percent)
aia_map.draw_quadrangle(coords, axes=ax,edgecolor="yellow",linestyle="-",linewidth=2,label='Region of Interest')
ax.plot_coord(intensity_coords, lw=1, color='magenta')
#ax.plot_coord(intensity_coords_above, lw=1, color='blue')
#ax.plot_coord(intensity_coords_below, lw=1, color='green')
ax.legend()
plt.show()
