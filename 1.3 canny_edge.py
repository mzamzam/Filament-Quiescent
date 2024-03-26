# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:34:54 2024

@author: WIN 11
"""

import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
from skimage import feature
import os
import sunpy.map
import pandas as pd

data_list = os.listdir('Data/AIA304/2012')
AIA_304 = f'Data/AIA304/2012/{data_list[160]}'
aia_map = sunpy.map.Map(AIA_304)
sun_rad = aia_map.meta['RSUN_OBS'] #in arcsec
sun_rad_ref = aia_map.meta['RSUN_REF']/1000 #in km
one_arcsec_to_km = sun_rad_ref/sun_rad

intensity = np.load(r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\intensity_along_line\intensity.npy")
png = r"C:\Users\WIN 11\Documents\PROJECTS\Filament Quiescent\Results\height-time_plot\fig_6b_polos.png"

orig_map = plt.colormaps.get('sdoaia304')
reversed_map = orig_map.reversed()

img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
img2 = feature.canny(img)
img3 = feature.canny(img, sigma=3)
img4 = cv2.Canny(img,30,150)
arr = feature.canny(intensity, sigma=11)

# plt.subplot(141),plt.imshow(img,cmap = 'binary')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(142),plt.imshow(edges,cmap = 'binary')
# plt.title('CV2'), plt.xticks([]), plt.yticks([])
# plt.subplot(143),plt.imshow(img2,cmap = 'binary')
# plt.title('skimage'), plt.xticks([]), plt.yticks([])
# plt.subplot(144),plt.imshow(img3,cmap = 'binary')
# plt.autoscale(False)
# plt.plot(coord[:,1], coord[:, 0], 'r.')
# plt.axis('off')
# plt.title('Peak local max'), plt.xticks([]), plt.yticks([])

fig, axes = plt.subplots(ncols=2, nrows=4,figsize=(4, 8))
ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes.flat

# plot orig-array
ax0.imshow(intensity, origin='lower', cmap='gray', aspect='auto')
ax0.set_title('Orig-array', fontsize=14)
# ax0.axis('off')

# plot orig-array
ax1.imshow(img)
ax1.set_title('Orig-png', fontsize=14)
# ax1.axis('off')

## Detect edges from array numpy with shape 182x307
coord = np.argwhere(arr==1)
ax2.imshow(arr, origin='lower', cmap='gray', aspect='auto')
ax2.set_title('CannySKI-array', fontsize=14)
# ax2.axis('off')

coord = coord[coord[:,0] > 50]
# coord = coord[coord[:,1] < 170]
coordy, index = np.unique(coord[:,0], return_index=True)
coord = coord[index]
df_coord = pd.DataFrame(coord, columns=['y', 'x'])
# df_coord.to_csv('Results/canny_edge/edge_coord.csv')
ax3.imshow(arr,origin='lower', cmap='binary', aspect='auto')
ax3.plot(coord[:, 1], coord[:, 0], color='red', marker='.',
         linestyle='None', markersize=2)
ax3.set_title('CannySKI-array', fontsize=14)
# ax3.axis('off')

## Detect edges from PNG using cannyCV2
coord2 = np.argwhere(img4!=0)
coord2 = coord2[coord2[:,0] < 500]
coord2 = coord2[coord2[:,0] > 70]
coord2 = coord2[coord2[:,1] > 50]
coordy2, index2 = np.unique(coord2[:,0], return_index=True)
coord2 = coord2[index2]

## cannyCV2 on png
ax4.imshow(img4)
ax4.set_title('CannyCV2-png', fontsize=14)
# ax4.axis('off')

## cannyCV2 on png
ax5.imshow(img4, cmap='binary')
ax5.plot(coord2[:, 1], coord2[:, 0], color='red', marker='.',
         linestyle='None', markersize=2)
ax5.set_title('CannyCV2-png', fontsize=14)
# ax5.axis('off')

## Detect edges from PNG using cannySKI
coord3 = np.argwhere(img2==1)
coord3 = coord3[coord3[:,0] < 500]
coord3 = coord3[coord3[:,0] > 70]
coord3 = coord3[coord3[:,1] > 50]
coordy3, index3 = np.unique(coord2[:,0], return_index=True)
coord3 = coord3[index3]

## cannySKI on png
ax6.imshow(img2)
ax6.set_title('CannySKI-png', fontsize=14)
# ax6.axis('off')

## cannySKI on png
ax7.imshow(img4, cmap='binary')
ax7.plot(coord2[:, 1], coord2[:, 0], color='red', marker='.',
         linestyle='None', markersize=2)
ax7.set_title('CannySKI-png', fontsize=14)
# ax7.axis('off')

# plt.tight_layout()
# plt.savefig('Results/canny_edge/fig_6c.png', dpi=200)
# plt.show()
# plt.close()

