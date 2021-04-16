import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import time


# gcc -c -I/usr/local/include/tblis/util  -I/usr/local/include/tblis tblis.c -o tblis.o -L/usr/local/lib/ -ltblis -lhptt  -march=native  -fopenmp
# gcc tblis.o -shared -I/usr/local/include/tblis/util  -I/usr/local/include/tblis -o libeincore.so -L/usr/local/lib/ -ltblis -lhptt  -march=native -fopenmp

lib = ctypes.CDLL("./libeincore.so")
_tensor_contract = lib.tensor_contract
_tensor_contract.restype = None

_tensor_contract.argtypes = [
    ndpointer(dtype=np.double, flags="C_CONTIGUOUS"),  # Data a
    ndpointer(ctypes.c_long),  # Shape a
    ndpointer(ctypes.c_int),  # Index a
    ctypes.c_int,  # Dimension a
    ndpointer(dtype=np.double, flags="C_CONTIGUOUS"),  # Data b
    ndpointer(ctypes.c_long),  # Shape b
    ndpointer(ctypes.c_int),  # Index b
    ctypes.c_int,  # Dimension b
    ndpointer(dtype=np.double, flags="C_CONTIGUOUS"),  # Data c
    ndpointer(ctypes.c_long),  # Shape c
    ndpointer(ctypes.c_int),  # Index c
    ctypes.c_int,  # Dimension c
]


def einsum(a, aIdx, b, bIdx, c, cIdx):

    a = a.astype(np.float64)
    aShape = np.asarray(a.shape, dtype=np.int64)
    aIdx = np.asarray(aIdx, dtype=np.int32)
    aDim = a.ndim

    b = b.astype(np.float64)
    bShape = np.asarray(b.shape, dtype=np.int64)
    bIdx = np.asarray(bIdx, dtype=np.int32)
    bDim = b.ndim

    c = c.astype(np.float64)
    cShape = np.asarray(c.shape, dtype=np.int64)
    cIdx = np.asarray(cIdx, dtype=np.int32)
    cDim = c.ndim
    # if   aIdx.flags['C_CONTIGUOUS']:

    start = time.time()
    _tensor_contract(
        a,
        aShape,
        aIdx,
        aDim,
        b,
        bShape,
        bIdx,
        bDim,
        c,
        cShape,
        cIdx,
        cDim,
    )
    finish = time.time()
    print(finish - start)
    return c


n=10000
k=10
m=10
j=100

a = np.random.rand(n, k, m).astype(np.float64)
b = np.random.rand(n, m, j).astype(np.float64)
c = np.zeros((n, k, j), dtype=np.float64)

einsum_string = "nkm,nmj->nkj"
left_strings, c_string = einsum_string.split("->")
a_string, b_string = left_strings.split(",")

a_idx = [ord(s) for s in list(a_string)]
b_idx = [ord(s) for s in list(b_string)]
c_idx = [ord(s) for s in list(c_string)]
print("Hitting einsums")
c = einsum(a, a_idx, b, b_idx, c, c_idx)
# print(c)
print("Numpy->")
start = time.time()
np.matmul( a, b)
# print(np.einsum(einsum_string, a, b))
finish = time.time()
print(finish - start)

# Broadcasted eincore matrix multiplication similar to numpy's MatMul function for tensors
# 	(n,k,m)*(n,m,j)->(n,k,j)
# 	Essentially does  km,mj->kj n-times matrix multiplications and stacks them.