""" 
To compile and link, move to /usr/local/lib 
- must have tblis installed 
- export LD_LIBRARY_PATH=/usr/local/lib

g++ -c -O3 -I/usr/local/include/tblis/util  -I/usr/local/include/tblis as_einsum.cxx -o as_einsum.o -L/usr/local/lib/ -ltblis -march=native  -fopenmp
g++ as_einsum.o -shared -I/usr/local/include/tblis/util  -I/usr/local/include/tblis -o libeinsum_tblis.so -L/usr/local/lib/ -ltblis   -march=native -fopenmp

"""
import sys
import re
import ctypes
import numpy as np
import time

libtblis = ctypes.CDLL("/usr/local/lib/libeinsum_tblis.so")

libtblis.as_einsum.restype = None
libtblis.as_einsum.argtypes = (
    np.ctypeslib.ndpointer(),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_int),
    np.ctypeslib.ndpointer(),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_int),
    np.ctypeslib.ndpointer(),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    np.ctypeslib.ndpointer(),
    np.ctypeslib.ndpointer(),
)

tblis_dtype = {
    np.dtype(np.float32): 0,
    np.dtype(np.double): 1,
    np.dtype(np.complex64): 2,
    np.dtype(np.complex128): 3,
}

np_einsum = np.einsum


def contract(subscripts, *tensors, **kwargs):
    """
    c = alpha * contract(a, b) + beta * c

    Args:
        tensors (list of ndarray) : Tensors for the operation.

    Kwargs:
        out (ndarray) : If provided, the calculation is done into this array.
        dtype (ndarray) : If provided, forces the calculation to use the data
            type specified.
        alpha (number) : Default is 1
        beta (number) :  Default is 0
    """
    sub_idx = re.split(",|->", subscripts)
    indices = "".join(sub_idx)
    c_dtype = kwargs.get("dtype", np.result_type(*tensors))
    if "..." in subscripts or not (
        np.issubdtype(c_dtype, np.float64) or np.issubdtype(c_dtype, np.complex)
    ):
        return np_einsum(subscripts, *tensors)

    alpha = kwargs.get("alpha", 1)
    beta = kwargs.get("beta", 0)
    c_dtype = np.result_type(c_dtype, alpha, beta)
    alpha = np.asarray(alpha, dtype=c_dtype)
    beta = np.asarray(beta, dtype=c_dtype)
    a = np.asarray(tensors[0], dtype=c_dtype)
    b = np.asarray(tensors[1], dtype=c_dtype)

    a_shape = a.shape
    b_shape = b.shape
    a_descr, b_descr, c_descr = sub_idx
    a_shape_dic = dict(zip(a_descr, a_shape))
    b_shape_dic = dict(zip(b_descr, b_shape))
    if any(
        a_shape_dic[x] != b_shape_dic[x] for x in set(a_descr).intersection(b_descr)
    ):
        raise ValueError(
            'operands dimension error for "%s" : %s %s' % (subscripts, a_shape, b_shape)
        )

    ab_shape_dic = a_shape_dic
    ab_shape_dic.update(b_shape_dic)
    c_shape = tuple([ab_shape_dic[x] for x in c_descr])

    out = kwargs.get("out", None)
    if out is None:
        order = kwargs.get("order", "C")
        c = np.empty(c_shape, dtype=c_dtype, order=order)
    else:
        assert out.dtype == c_dtype
        assert out.shape == c_shape
        c = out

    a_shape = (ctypes.c_size_t * a.ndim)(*a_shape)
    b_shape = (ctypes.c_size_t * b.ndim)(*b_shape)
    c_shape = (ctypes.c_size_t * c.ndim)(*c_shape)

    nbytes = c_dtype.itemsize
    a_strides = (ctypes.c_size_t * a.ndim)(*[x // nbytes for x in a.strides])
    b_strides = (ctypes.c_size_t * b.ndim)(*[x // nbytes for x in b.strides])
    c_strides = (ctypes.c_size_t * c.ndim)(*[x // nbytes for x in c.strides])

    a_descr_int = [ord(s) for s in a_descr]
    b_descr_int = [ord(s) for s in b_descr]
    c_descr_int = [ord(s) for s in c_descr]

    libtblis.as_einsum(
        a,
        a.ndim,
        a_shape,
        a_strides,
        (ctypes.c_int * len(a_descr_int))(*a_descr_int),
        b,
        b.ndim,
        b_shape,
        b_strides,
        (ctypes.c_int * len(b_descr_int))(*b_descr_int),
        c,
        c.ndim,
        c_shape,
        c_strides,
        (ctypes.c_int * len(c_descr_int))(*c_descr_int),
        tblis_dtype[c_dtype],
        alpha,
        beta,
    )
    return c
