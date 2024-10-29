import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    eigenvector = np.random.rand(data.shape[1])

    for _ in range(num_steps):
        eigenvector1 = np.dot(data, eigenvector)

        eigenvector1_norm = np.linalg.norm(eigenvector1)

        eigenvector = eigenvector1 / eigenvector1_norm

    eigenvalue = float(np.mean(data.dot(eigenvector.T)/eigenvector))

    return eigenvalue, eigenvector