try:
    from scipy.sparse import csr_array   # requires scipy version > 1.7.3 (check)
except ImportError as e:
    from scipy.sparse import csr_matrix as csr_array   # let's try this first
