import numpy as np

def uniform_intervals(a, b, n):
    
    return np.linspace(a, b, n)



def cyclic123_array(n):
    
    base_array = np.array([1, 2, 3])
    return np.tile(base_array, n)



def first_n_odd_number(n):
    
    return np.arange(1, 2*n, 2)



def zeros_array_with_border(n):
   
    Z = np.zeros((n,n))
    Z[0,:] = 1
    Z[-1,:] = 1
    Z[:,0] = 1
    Z[:,-1] = 1
    return Z



def chess_board(n):
    

    x = np.zeros((n,n),dtype=int)
    x[1::2,::2] = 1
    x[::2,1::2] = 1
    return x



def matrix_with_sum_index(n):
    
    rows = np.arange(n)
    cols = np.arange(n)
    return rows[:, np.newaxis] + cols



def cos_sin_as_two_rows(a, b, dx):
    
    x = np.arange(a, b, dx)
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    return np.array([cos_x, sin_x])



def compute_mean_rowssum_columnssum(A):
    
    mean = np.mean(A)
    rows_sum = np.sum(A, axis=1)
    columns_sum = np.sum(A, axis=0)
    return mean, rows_sum, columns_sum



def sort_array_by_column(A, j):
    
    return A[A[:,j].argsort()]



def compute_integral(a, b, f, dx, method):
   
    x = np.arange(a, b, dx)
    if method == 'rectangular':
        return np.sum(f(x) * dx)
    elif method == 'trapezoidal':
        return (dx/2) * (f(a) + 2*np.sum(f(x[1:])) + f(b-dx))
    elif method == 'simpson':
        n = len(x)
        if n % 2 == 0:
            x = x[:-1] # make n odd
            n = len(x)
        h = (b - a) / (n - 1) # redefine h
        return (h/3) * (f(a) + 4*np.sum(f(x[1::2])) + 2*np.sum(f(x[2:-1:2])) + f(b-dx))


