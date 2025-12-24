import numpy as np


# Inverse Distance Weighting method for interpolation 
# but this is not working well for extrapolation ....
# What todo... what to do...
def intpND(X_arr_list, Y_arr, x_targ_arr, order=1):
    N = len(X_arr_list) # Number of data
    w_list = []
    for xx_da in X_arr_list:
        d_tmp = np.sqrt( np.sum((x_targ_arr - xx_da)**2) )
        w_list.append(1/(d_tmp+0.000001)**order)
    w_sum = np.sum(w_list)
    Y_pred = 0
    for ww, yy in zip(w_list, Y_arr):
        Y_pred = Y_pred + ww*yy/w_sum
    return Y_pred

# Test for intpND

if __name__== '__main__':
    # Suppose we have 3 points
    x_1 = np.array([1, 1,])
    x_2 = np.array([2, 2,])
    Y_test = np.array([10,20,])
    X_data = [x_1, x_2,]
    Y_test = intpND(X_data, Y_test, np.array([1.5, 1.5]))
    print(Y_test)
    
    x1 = np.array([1, 1,1])
    x2 = np.array([1, 1, 2])
    x3 = np.array([1, 2, 1])
    x4 = np.array([2, 1, 1])
    x5 = np.array([1, 2, 2])
    x6 = np.array([2, 1, 2])
    x7 = np.array([2, 2, 1])
    x8 = np.array([2, 2, 2])

    X_data = [x1,x2,x3,x4,x5,x6,x7,x8,]
    Y_data2 = [1, 1, 1, 2, 1, 2, 2, 2, 2]
    # Only a function of x axis data
    x_test2 = np.array([1.5, 1.2, 1.9])
    Y_test2 = intpND(X_data, Y_data2, x_test2)
    print(Y_test2)

