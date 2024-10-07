# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 17:23:46 2024

@author: MZN
"""

import requests
from bs4 import BeautifulSoup
import os

y = 2023
m = 4
d = 20 # 9, 10, 11, 12
# H = 0 # 0 s.d. 23 
typ= 'EUI304'

def zfill(x):
    return str(x).zfill(2)

#https://sidc.be/EUI/data/L2/2023/04/20/solo_L2_eui-fsi304-image_20230420T150020245_V01.fits
for H in range(0,21,1):
    tanggal_eui = f'{y}/{zfill(m)}/{zfill(d)}/'
    eui_url = f'https://sidc.be/EUI/data/L2/{tanggal_eui}'
    page = requests.get(eui_url).text
    soup = BeautifulSoup(page, 'html.parser')
    eui_data = [eui_url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('.fits')]
    n = len(eui_data)
    # i = n-1
    os.makedirs(f'Data/{typ}/{tanggal_eui[:4]}', exist_ok=True)
    data_dir = f'Data/{typ}/{tanggal_eui[:4]}'
    for i in range(n-1):
        f = open(f'{data_dir}/{eui_data[i][-26:]}', 'wb')
        print(f'Downloading {eui_data[i][-26:]} data ...')
        response = requests.get(eui_data[i])
        f.write(response.content)
        f.close()

### Failed. the filenames are similar, so it is quite hard to distinguish them.