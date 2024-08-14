"""
To download only AIA data automatically.
The data is saved in a new 'Data' directory.
Change y, m, d, typ, and H based on your preference.
"""

import requests
from bs4 import BeautifulSoup
import os

y = 2012
m = 3
d = 12
typ= 'AIA304'
def zfill(x):
    return str(x).zfill(2)
## data source: http://jsoc2.stanford.edu/data/aia/synoptic/2012/03/09/H0000/AIA20120309_0000_0304.fits
for H in range(0,4,1):   ## H means hours, 0 is counted, 4 is not counted, 1 is step
    tanggal_aia = f'{y}/{zfill(m)}/{zfill(d)}/H{zfill(H)}00'
    aia_url = f'http://jsoc2.stanford.edu/data/aia/synoptic/{tanggal_aia}'
    page = requests.get(aia_url).text
    soup = BeautifulSoup(page, 'html.parser')
    aia_data = [aia_url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('304.fits')]
    n = len(aia_data)
    # i = n-1
    os.makedirs(f'Data/{typ}/{tanggal_aia[:4]}/{zfill(m)}', exist_ok=True)
    data_dir = f'Data/{typ}/{tanggal_aia[:4]}/{zfill(m)}'
    for i in range(n-1):
        f = open(f'{data_dir}/{aia_data[i][-26:]}', 'wb')
        print(f'Downloading {aia_data[i][-26:]} data ...')
        response = requests.get(aia_data[i])
        f.write(response.content)
        f.close()

