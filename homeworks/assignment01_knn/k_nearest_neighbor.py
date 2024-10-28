import numpy as np

class KNearestNeighbor:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                d = np.sqrt(((X[i]-self.X_train[j])**2).sum())
                dists[i, j] = d

        return dists

    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            d = np.sqrt(np.sum((X[i] - self.X_train)**2, axis=1))
            dists[i] = d

        return dists

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        s_train = np.sum(self.X_train**2, axis=1, keepdims=True)
        s_test = np.sum(X**2, axis=1, keepdims=True)

        dists = np.sqrt(-2 * self.X_train.dot(X.T) + s_train + s_test.T).T

        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            cloest_y = self.y_train[np.argsort(dists[i,:])[:k]]

            y_pred[i] = np.argmax(np.bincount(cloest_y))

        return y_pred
