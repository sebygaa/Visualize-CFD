import idwpack_v01 as ip
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.interpolate import griddata


df_coor = pd.read_excel('coords.xlsx')
df_Tdata = pd.read_excel('T_353K_4ms.xlsx')
print('\n'*5)
#print(df_coor.iloc[0:3,:])

x_da = df_coor.iloc[:,1].to_numpy()
y_da = df_coor.iloc[:,2].to_numpy()
z_da = df_coor.iloc[:,3].to_numpy()

T_da = df_Tdata.iloc[:,2].to_numpy()

##################
y_set = 0.1
##################

# Find y <= 0.11 & y >= 0.09
arg_y_set = (y_da <= y_set+0.08) & (y_da >= y_set - 0.08)

# Refined data
x_re = x_da[arg_y_set]
y_re = y_da[arg_y_set]
z_re = z_da[arg_y_set]

T_re = T_da[arg_y_set]

#print(f"Filtered data length: {len(x_re)}")
#print(f"Sample x_re: {x_re[:5]}")
#print(f"Sample y_re: {y_re[:5]}")
#print(f"Sample z_re: {z_re[:5]}")

x_ran = np.linspace(x_re.min(), x_re.max(), 81)
z_ran = np.linspace(z_re.min(), z_re.max(), 81)


T_intp_list = []
for xx in x_ran:
    T_tmp = []
    for zz in z_ran:
        dx = x_re - xx
        dy = y_re - y_set
        dz = z_re - zz
        rr = np.sqrt(dx**2 + dy**2 + dz**2)
        arg_sort = np.argsort(rr)
     
        r_sort = rr[arg_sort]
        T_sort = T_re[arg_sort]
        x_sort = x_re[arg_sort]
        y_sort = y_re[arg_sort]
        z_sort = z_re[arg_sort]

        r_use = r_sort[0:8]
        T_use = T_sort[0:8]
        x_use = x_sort[0:8]
        y_use = y_sort[0:8]
        z_use = z_sort[0:8]
     
        X_coord = np.vstack([x_use,
                             y_use,
                             z_use, ],).T
        
        X_targ = np.array([xx, y_set, zz])
        #print(f"X_coord shape: {X_coord.shape}, unique points: {len(np.unique(X_coord, axis=0))}")
        
        T = ip.idw_auto(X_coord, T_use, X_targ, power=1)
        #print(f"x={xx:.4f}, y={y_set:.4f}, z={zz:.4f} => T={T:.4f}")

        T_tmp.append(T)
    T_intp_list.append(T_tmp)
T_intp = np.asarray(T_intp_list)

plt.figure()
levels = np.linspace(290, 400,111)
# contour = plt.contourf(X, Z, T, levels=15, cmap='RdBu_r')
contour = plt.contourf(x_ran, z_ran, T_intp, cmap='turbo',
                       vmin = 300,
                       vmax = 385,
                       levels = 40)
plt.colorbar(contour, label='Temperature (K)')
#plt.scatter(x_re, z_re, color='black', s=10, alpha=0.5, label='Original Data Points')

plt.title('2D Contour Plot from Scattered Data')
plt.xlabel('X axis (m)')
plt.ylabel('Z axis (m)')
#plt.legend()
plt.show()


