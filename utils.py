#utils
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags

def make_adjacency(rf, X):
    leafIDs = rf.apply(X)
    n_samples, n_trees = leafIDs.shape
    max_leaf = leafIDs.max()
    # give each (tree, leaf) a unique global ID
    leafIDs_global = (leafIDs + np.arange(n_trees) * max_leaf).astype(int)

    # Build sparse membership matrix M
    row_ind = np.repeat(np.arange(n_samples), n_trees)
    col_ind = leafIDs_global.ravel()
    data = np.ones_like(row_ind, dtype=np.float32)

    M = csr_matrix((data, (row_ind, col_ind)), shape=(n_samples, col_ind.max() + 1))

    # leaf sizes and weights
    leaf_sizes = np.array(M.sum(axis=0)).ravel()
    leaf_weights = 1.0 / np.maximum(leaf_sizes, 1)  # avoid div by zero

    M_norm = M @ diags(leaf_weights)
    A = (M_norm @ M.T) / n_trees
    
    metadata = {
        "leaf_weights": leaf_weights,
        "leafIDs": leafIDs,
        'n_trees': n_trees,
        'max_leaf': max_leaf,
        'M_train': M
    }
    return A.toarray(), metadata

def new_adjacency(rf, X_new, metadata):
    M_train = metadata["M_train"]
    leaf_weights = metadata["leaf_weights"]
    n_trees = metadata["n_trees"]
    max_leaf = metadata["max_leaf"]
    
    n_new = X_new.shape[0]
    
    # Get terminal nodes for new data
    leafIDs_new = rf.apply(X_new)
    leafIDs_global_new = (leafIDs_new + np.arange(n_trees) * max_leaf).astype(int)
    
    # Merge train and new leaves
    leafIDs_global_train = M_train.indices
    leafIDs_union = np.union1d(leafIDs_global_train, leafIDs_global_new.ravel())
    
    # Map global IDs to columns
    id_map_train = {lid: i for i, lid in enumerate(leafIDs_union)}
    id_map_new = {lid: i for i, lid in enumerate(leafIDs_union)}
    
    # Sparse matrices
    row_ind_train = np.repeat(np.arange(M_train.shape[0]), n_trees)
    col_ind_train = np.array([id_map_train[lid] for lid in M_train.indices])
    M_train_sparse = csr_matrix((np.ones_like(row_ind_train), 
                                 (row_ind_train, col_ind_train)),
                                shape=(M_train.shape[0], len(leafIDs_union)))
    
    row_ind_new = np.repeat(np.arange(n_new), n_trees)
    col_ind_new = np.array([id_map_new[lid] for lid in leafIDs_global_new.ravel()])
    M_new_sparse = csr_matrix((np.ones_like(row_ind_new), 
                               (row_ind_new, col_ind_new)),
                              shape=(n_new, len(leafIDs_union)))
    
    # Normalize new matrix
    leaf_sizes = np.array(M_train_sparse.sum(axis=0)).ravel()
    leaf_weights = 1.0 / leaf_sizes
    leaf_weights[~np.isfinite(leaf_weights)] = 0
    M_new_norm = M_new_sparse @ diags(leaf_weights)
    
    # Weighted adjacency for new data vs training
    A_new = (M_train_sparse @ M_new_norm.T).T / n_trees
    return A_new.toarray()

def toeplitz(d, rho):
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def simulate_data(n=1000, d=10, rho=0.9, beta=None, seed=0):
    """
    Simulate data:
      X ~ N(0, Σ),  Σ_ij = rho^{|i-j|},
      Y ~ Bernoulli( sigmoid(X @ beta) ).
    """
    rng = np.random.default_rng(seed)
    sigma = toeplitz(d, rho)

    # default beta if not provided
    if beta is None:
        beta = np.concatenate([np.zeros(d // 2), np.ones(d // 2)])
    beta = np.asarray(beta)

    # sample X
    L = np.linalg.cholesky(sigma)
    Z = rng.standard_normal((n, d))
    X = Z @ L.T

    # sample Y
    logits = X @ beta
    p = sigmoid(logits)
    y = rng.binomial(1, p)

    return X, y, beta, sigma