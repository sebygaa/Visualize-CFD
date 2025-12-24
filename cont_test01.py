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

# Find y <= 0.11 & y >= 0.09
y_set = (y_da <= 0.11) & (y_da >=0.09) & (x_da >=0.1) & (x_da <= 1.05)

# Refined data
x_re = x_da[y_set]
y_re = y_da[y_set]
z_re = z_da[y_set]

T_re = T_da[y_set]

print(x_da)
print(type(x_da))

# Check the point position
fig = plt.figure() 
ax = fig.add_subplot(projection = '3d')

ax.scatter(x_da,y_da,z_da, marker = 'o')
#ax.scatter(x_re, y_re, z_re, marker = 'o')
ax.set_xlabel('x-axis')
ax.set_ylabel('z-axis')
plt.show()


# try using tircontourf 
plt.figure()
levels = np.linspace(310, 370, 30)
contour = plt.tricontourf(x_re, z_re, T_re,
                          levels = 25, 
                          vmin = 310,
                          vmax = 360,
                          cmap='RdBu_r')

plt.colorbar(contour, label='Value (Z)',)

plt.show()
# Gridize the data

xi = np.linspace(x_re.min(), x_re.max(), 200)
yi = np.linspace(z_re.min(), z_re.max(), 200)
X, Z = np.meshgrid(xi, yi)

# 'cubic' is smooth; 'linear' is faster; 'nearest' is blocky
T = griddata((x_re, z_re), T_re, (X, Z), method='cubic')

plt.figure()
#plt.contourf([x_re, z_re], T_re,)
levels = np.linspace(310, 370,31)
# contour = plt.contourf(X, Z, T, levels=15, cmap='RdBu_r')
contour = plt.contourf(X, Z, T, cmap='turbo',
                       vmin = 310,
                       vmax = 370,
                       levels = 25)
plt.colorbar(contour, label='Temperature (K)')
#plt.scatter(x_re, z_re, color='black', s=10, alpha=0.5, label='Original Data Points')

plt.title('2D Contour Plot from Scattered Data')
plt.xlabel('X axis (m)')
plt.ylabel('Z axis (m)')
#plt.legend()
plt.show()
