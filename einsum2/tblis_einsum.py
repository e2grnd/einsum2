"""
A Python interface to mimic numpy.einsum
"""

# g++ -c -O3 -I/usr/local/include/tblis/util  -I/usr/local/include/tblis as_einsum.cxx -o as_einsum.o -L/usr/local/lib/ -ltblis -march=native  -fopenmp
# g++ as_einsum.o -shared -I/usr/local/include/tblis/util  -I/usr/local/include/tblis -o libeinsum_tblis.so -L/usr/local/lib/ -ltblis   -march=native -fopenmp

import sys
import re
import ctypes
import numpy as np
import time

libtblis = ctypes.CDLL("./libeinsum_tblis.so")

libtblis.as_einsum.restype = None
libtblis.as_einsum.argtypes = (
    np.ctypeslib.ndpointer(),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    np.ctypeslib.ndpointer(),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    np.ctypeslib.ndpointer(),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
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


def _contract(subscripts, *tensors, **kwargs):
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

    if "->" not in subscripts:
        # Find chararacters which appear only once in the subscripts for c_descr
        for x in set(indices):
            if indices.count(x) > 1:
                indices = indices.replace(x, "")
        sub_idx += [indices]

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

    libtblis.as_einsum(
        a,
        a.ndim,
        a_shape,
        a_strides,
        a_descr.encode("ascii"),
        b,
        b.ndim,
        b_shape,
        b_strides,
        b_descr.encode("ascii"),
        c,
        c.ndim,
        c_shape,
        c_strides,
        c_descr.encode("ascii"),
        tblis_dtype[c_dtype],
        alpha,
        beta,
    )
    return c


def einsum(subscripts, *tensors, **kwargs):
    subscripts = subscripts.replace(" ", "")
    if len(tensors) <= 1:
        out = np_einsum(subscripts, *tensors, **kwargs)
    elif len(tensors) <= 2:
        out = _contract(subscripts, *tensors, **kwargs)
    else:
        sub_idx = subscripts.split(",", 2)
        res_idx = "".join(set(sub_idx[0] + sub_idx[1]).intersection(sub_idx[2]))
        res_idx = res_idx.replace(",", "")
        script0 = sub_idx[0] + "," + sub_idx[1] + "->" + res_idx
        subscripts = res_idx + "," + sub_idx[2]
        tensors = [_contract(script0, *tensors[:2])] + list(tensors[2:])
        out = einsum(subscripts, *tensors, **kwargs)
    return out


if __name__ == "__main__":

    i, j, k, l = 500, 100, 200, 200
    a = np.random.rand(i, j, k)
    b = np.random.rand(i, k, l)
    subscripts = "ijk,ikl->ijl"

    print("Arrays generated for eincore..")

    start = time.time()
    d = einsum(subscripts, a, b)
    print("Tblis entry: ", d.flatten()[0])
    finish = time.time()
    tblis = finish - start
    print("tblis", tblis)

    start = time.time()
    d = np.einsum(subscripts, a, b)
    print("NPY entry: ", d.flatten()[0])
    finish = time.time()
    npy = finish - start
    print("npy", npy)

    print("Speedup: ", npy / tblis)

    size = 2
    a = np.random.rand(size, size, size, size, size, size, size)
    b = np.random.rand(size, size, size, size, size, size, size)
    subscripts = "abcdefg,hijklmn->abcdefghijklmn"

    print("\n\nArrays generated for big product..")

    start = time.time()
    d = einsum(subscripts, a, b)
    print("Tblis entry: ", d.flatten()[0])
    finish = time.time()
    tblis = finish - start
    print("tblis", tblis)

    start = time.time()
    d = np.einsum(subscripts, a, b)
    print("NPY entry: ", d.flatten()[0])
    finish = time.time()
    npy = finish - start
    print("npy", npy)

    print("Speedup: ", npy / tblis)
