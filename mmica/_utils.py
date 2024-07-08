# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#
# License: MIT
import os

import numpy as np

from ._conjugate_gradient import cg_c as cython_cg_c

def python_cg_c_with_extra_info(B: np.ndarray, i: int, max_iter: int, tol: float, N: int):

    x = np.zeros(N)
    B_diag = np.empty(N)
    r = np.empty(N)
    y = np.empty(N)
    p = np.empty(N)
    Ap = np.empty(N)
    norms = np.empty(max_iter) * np.nan

    B_diag = np.diag(B)
    r = B[i,:]
    inv_bii = 1. / B[i, i]
    r = r * inv_bii
    x[i] = inv_bii
    r[i] = 0.
    for j in range(N):
        y[j] = r[j] / B_diag[j]
    p = y
    prod_old = r @ y
    for n in range(max_iter):
        Ap = B @ p
        alpha = prod_old / (p @ Ap)
        minus_alpha = - alpha
        x = x +  minus_alpha * p
        r = r + Ap * minus_alpha
        norm = np.linalg.norm(r)
        norms[n] = norm
        if norm < tol:
            break
        for j in range(N):
            y[j] = r[j] / B_diag[j]
        prod_new = r@y
        beta = prod_new / prod_old
        prod_old = prod_new
        for j in range(N):
            p[j] = y[j] + beta * p[j]
    return x, n, norms

def python_cg_c(B: np.ndarray, i: int, max_iter: int, tol: float, N: int):
    return python_cg_c_with_n(B, i, max_iter, tol, N)[0]


def amari_d(W, A):
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)
    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


def whitening(Y, mode='sph'):
    '''
    Whitens the data Y using sphering or pca
    '''
    R = np.dot(Y, Y.T) / Y.shape[1]
    U, D, _ = np.linalg.svd(R)
    if mode == 'pca':
        W = U.T / np.sqrt(D)[:, None]
        Z = np.dot(W, Y)
    elif mode == 'sph':
        W = np.dot(U, U.T / np.sqrt(D)[:, None])
        Z = np.dot(W, Y)
    return Z, W


def cg(B, i, max_iter=10, tol=1e-10, use_cython=True):
    '''
    Wrapper to call Cython
    '''
    cg_c = cython_cg_c
    if not use_cython:
        cg_c = python_cg_c
    return cg_c(B, i, max_iter, tol, B.shape[0])