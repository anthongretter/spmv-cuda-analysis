import numpy as np
from scipy.io import mmwrite
from scipy.sparse import coo_matrix

def generate_concentrated_matrix(n_rows=19000, n_cols=19000,
                                 heavy_rows_ratio=0.01,  # 1% of rows
                                 heavy_row_nnz=10000,    # very dense rows
                                 normal_row_nnz=10):     # sparse rows
    np.random.seed(42)  # reproducibility
    data = []
    rows = []
    cols = []

    num_heavy_rows = int(n_rows * heavy_rows_ratio)
    heavy_row_indices = np.random.choice(n_rows, num_heavy_rows, replace=False)

    for row in range(n_rows):
        if row in heavy_row_indices:
            nnz = min(n_cols, heavy_row_nnz)
        else:
            nnz = min(n_cols, normal_row_nnz)

        col_indices = np.random.choice(n_cols, nnz, replace=False)
        values = np.random.rand(nnz).astype(np.float32)

        rows.extend([row] * nnz)
        cols.extend(col_indices)
        data.extend(values)

    coo = coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    mmwrite("concentrated_matrix.mtx", coo)
    print(f"Matrix saved to concentrated_matrix.mtx with shape {n_rows}x{n_cols} and {len(data)} non-zeros.")

if __name__ == "__main__":
    generate_concentrated_matrix()
