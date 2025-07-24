import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite
import random

def generate_irregular_sparse_matrix(n_rows=19000, n_cols=19000, min_nnz=0, max_nnz=300, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    data = []
    rows = []
    cols = []

    for i in range(n_rows):
        nnz_in_row = random.randint(min_nnz, max_nnz)
        col_indices = np.random.choice(n_cols, nnz_in_row, replace=False)
        values = np.random.rand(nnz_in_row).astype(np.float32)

        rows.extend([i] * nnz_in_row)
        cols.extend(col_indices)
        data.extend(values)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    return A

if __name__ == "__main__":
    matrix = generate_irregular_sparse_matrix()
    mmwrite("irregular_19000x19000.mtx", matrix)
    print("Saved: irregular_19000x19000.mtx")
