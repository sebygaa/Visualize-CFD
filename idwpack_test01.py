import idwpack_v01 as ip
import numpy as np

X_data = np.array([
    [1,1,1,],
    [1,1,2],
    [1,2,1],
    [2,1,1],
    [1,2,2,],
    [2,1,2],
    [2,2,1],
    [2,2,2,],
    ], dtype=float)
Y_data =  np.array([1, 1, 1, 2, 1, 2, 2, 2,])
x_targ1 = np.array([1.5, 1.2, 1.9])
x_targ2 = np.array([2.5, 1.2, 1.9])

y_targ1 = ip.idw_auto(X_data, Y_data, x_targ1, power = 1)
print(y_targ1)
y_targ2 = ip.idw_auto(X_data, Y_data, x_targ2, power = 1)
print(y_targ2)

