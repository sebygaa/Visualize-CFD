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

# ChatGPT did let me know how to deal with 
# W = diag(w1,w2,w3,w4 ...)
# C = np.mat( [[1,1,1, ... , 1],
#              [x_1, x_2, x_3, ...],
#              [y_1, y_2, y_3, ...],
#              [z_1, z_2, z_3, ...], )
# c = np.mat([[1],
#             [x_targ,],
#             [y_targ,],
#             [z_targ,],]   )
# Then... ChatGPT said 
# lambda = W * C.T * np.linalg.inv(C * W * C.T)*c


def intpND_extra(X_arr_list, Y_arr, x_targ_arr, order=1):
    N = len(X_arr_list) # Number of data
    w_list = []
    C_list = []
    for xx_da in X_arr_list:
        #d_tmp = np.sqrt( np.sum((x_targ_arr - xx_da)**2) )
        d_tmp = x_targ_arr - xx_da
        w_list.append(1/(d_tmp+0.0000001)**order)
        C_list.append(np.concatenate( [ [1],xx_da],))
        #print(C_list)

    w_sum = np.sum(w_list)
    Y_pred = 0

    W_mat = np.diag(w_list)
    C_mat = np.matrix(C_list).T 
    c = np.matrix(np.concatenate( [[1,],x_targ_arr] ) ).T
    # print('W_mat :',W_mat.shape)
    # print('C_mat :',C_mat.shape)
    # print('c  : ', c.shape)

    lambd_mat = W_mat @ C_mat.T @ np.linalg.inv(C_mat @ W_mat @ C_mat.T)@c
    Y_targ = np.sum(np.array(lambd_mat)*Y_arr)
    print('Sum = ', np.sum(lambd_mat))
    return Y_targ

# Test for intpND

if __name__== '__main__':
    # Suppose we have 3 points
    x_1 = np.array([1, 1,])
    x_2 = np.array([2, 2,])
    Y_test = np.array([10,20,])
    X_data = [x_1, x_2,]
    Y_test = intpND(X_data, Y_test, np.array([1.5, 1.5]))
    print(Y_test)
    
    x1 = np.array([1, 1, 1])
    x2 = np.array([1, 1, 2])
    x3 = np.array([1, 2, 1])
    x4 = np.array([2, 1, 1])
    x5 = np.array([1, 2, 2])
    x6 = np.array([2, 1, 2])
    x7 = np.array([2, 2, 1])
    x8 = np.array([2, 2, 2])

    X_data = [x1,x2,x3,x4,x5,x6,x7,x8,]
    Y_data2 = [1, 1, 1, 2, 1, 2, 2, 2]
    # Only a function of x axis data
    x_test2 = np.array([1.5, 1.2, 1.50])
    Y_test2 = intpND(X_data, Y_data2, x_test2)
    print(Y_test2)
    
    y_test3 = intpND_extra(X_data, Y_data2, x_test2)
    print(y_test3)

    
    
    


