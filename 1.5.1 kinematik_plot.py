import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from skimage import feature
from datetime import timedelta

# --- Configuration ---
dtdt = '20230420'
sig = 9 # Sigma for Canny edge detection

# Load your pre-processed data
intensity = np.load(f"Results/intensity_along_line/intensity_{dtdt}.npy")
dt_intensity = pd.read_csv(f"Results/intensity_along_line/datetime_intensity_{dtdt}.csv",
                           parse_dates=['date'])
ang_sep = pd.read_csv(f"Results/intensity_along_line/ang_sep_{dtdt}.csv")

# Setup time and height arrays
dt_intensity['minute'] = (dt_intensity['date'] - dt_intensity['date'][0]).dt.total_seconds() / 60
start_time = dt_intensity['date'].iloc[0]

# --- Processing ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from skimage import feature
from datetime import timedelta

# ... [Keep your existing data loading code for intensity, dt_intensity, and ang_sep] ...

# 1. Edge Detection
arr = feature.canny(intensity, sigma=9)
coord = np.argwhere(arr == 1)
df_coord = pd.DataFrame(coord, columns=['y', 'x'])

# 2. Map coordinates
df_coord['y_arcsec'] = df_coord['y'].apply(lambda i: ang_sep.loc[i].h_arcsec)
df_coord['x_dt'] = df_coord['x'].apply(lambda i: start_time + timedelta(minutes=dt_intensity.loc[i].minute))

# 3. REVISION: Limit the time range to exclude the noisy right side
# Adjust '15:15' based on your specific vertical dashed line location
cutoff_time = start_time + timedelta(hours=8, minutes=15)
df_coord_filtered = df_coord[df_coord['x_dt'] <= cutoff_time].copy()

# 4. Extract Leading Edge from the filtered data
df_leading_edge = df_coord_filtered.sort_values('y_arcsec', ascending=False).drop_duplicates('x')
df_leading_edge = df_leading_edge.sort_values('x_dt')

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 5))

# Plot all cyan edges as background
ax.plot(df_coord['x_dt'], df_coord['y_arcsec'], color='cyan',
        marker='.', linestyle='None', markersize=1, alpha=0.3)

# Plot ONLY the cleaned leading edge in red
ax.plot(df_leading_edge['x_dt'], df_leading_edge['y_arcsec'], color='red',
        linestyle='-', linewidth=1.5)

# Formatting
ax.set_ylim(0, 500)
ax.set_ylabel('Height (arcsec)', fontsize=14)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.tick_params(direction='in', top=True, right=True, which='both')

plt.show()