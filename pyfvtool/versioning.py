import numpy as np

try:
    from scipy.sparse import csr_array   # requires scipy version > 1.7.3 (check)
except ImportError as e:
    from scipy.sparse import csr_matrix as csr_array   # let's try this first

    
def sub2ind(array_shape, rows, cols):
    # # ravel_index and unravel_index are much better solutions here for speed reasons
    # ind = rows*array_shape[1] + cols
    # ind[ind < 0] = -1
    # ind[ind >= array_shape[0]*array_shape[1]] = -1
    # return ind
    #
    # Use Fortran-style (column-major) ordering and return the index corresponding to the input position in the flattened array for multiple indices
    # return np.ravel_multi_index((rows, cols), array_shape, mode='raise', order='F')
    return np.ravel_multi_index((rows, cols), array_shape, mode='raise', order='C')

def ind2sub(array_shape, ind):
    # # ravel_index and unravel_index are much better solutions here for speed reasons
    # ind[ind < 0] = -1
    # ind[ind >= array_shape[0]*array_shape[1]] = -1
    # rows = (ind.astype('int') / array_shape[1])
    # cols = ind % array_shape[1]
    # return (rows, cols)
    #
    # Converts a flat index of array of flat indices into a tuple of coordinate arrays
    # return np.unravel_index(ind, array_shape, order='F')
    return np.unravel_index(ind, array_shape, order='C')

