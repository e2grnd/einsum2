from eincore import einsum
import numpy as np
import time
import random
from random import sample
from math import floor
import opt_einsum as oe


def random_einsum_test(dim=2, min_size=1, max_size=10, union_flag=False):
    einsum_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    einsum_chars_list = list(einsum_chars[:dim])
    shapes = {}
    for s in einsum_chars:
        shapes[s] = random.randint(min_size, max_size)

    a_string = "".join(sample(einsum_chars_list, dim))
    b_string = "".join(sample(einsum_chars_list, dim))
    if union_flag:
        c_string = "".join(set(a_string).union(b_string))
    else:
        c_string = ""

    einsum_string = a_string + "," + b_string + "->" + c_string[: floor(dim / 2)]
    a_sizes, b_sizes = [shapes[s] for s in a_string], [shapes[s] for s in b_string]
    a, b = np.random.rand(*a_sizes), np.random.rand(*b_sizes)

    print("shape and size a: ", a_sizes, a.size)
    print("shape and size b: ", b_sizes, b.size)
    return einsum_string, a, b


def tensorGen(einsumStr, N, M):
    splitArrowEinsumStr = einsumStr.split("->")
    leftStrList = splitArrowEinsumStr[0].split(",")
    shapes = {}
    for aStr in leftStrList:
        for i, r in enumerate(aStr):
            try:
                shapes[r]
            except:
                if i % 2 == 0:
                    shapes[r] = N
                else:
                    shapes[r] = M

    tensors = []
    for aStr in leftStrList:
        dims = []
        size = 1
        for r in aStr:
            dims.append(shapes[r])
            size *= shapes[r]
        a = np.random.rand(size).reshape(dims)
        # a = np.array(np.arange(size), dtype=np.float64).reshape(dims)
        tensors.append(a)

    return tensors


# subscripts = "A,A,ji^,j,igh,g`fC,`^_,^,_,f,CAB,B,B,h,XWQUV,USTCQRab,Q,R,T,V,POD,P,OMNFLC,FDEba,D,E,LNed^,edbc,b`aC,a,c,|{k,|,{yz,ymxC,mkl,k,l,x,z,KJD,K,JHI,HFGC,G,I,wvk,w,vtu,abdcf,fcab,tmsC,s,u,]京Q,],京Z[,ZSYC,Y,[,rqk,r,qop,omnC,n,p->A"
# tensors = tensorGen(subscripts, 8, 10)


# print("Arrays generated for eincore...")

# start = time.time()
# d = oe.contract(subscripts, *tensors, optimize="random-greedy", backend="numpy")
# print("Numpy entry: ", d.flatten()[1])
# finish = time.time()
# Numpy_time = finish - start
# print("Numpy time: ", Numpy_time)

# start = time.time()
# d = oe.contract(subscripts, *tensors, optimize="random-greedy", backend="eincore")
# print("Eincore entry: ", d.flatten()[1])
# finish = time.time()
# Eincore_time = finish - start
# print("Eincore time: ", Eincore_time)
# print("Speedup: ", Numpy_time / Eincore_time)


subscripts, a, b = random_einsum_test(dim=8, min_size=10, max_size=11, union_flag=True)
print(subscripts)

print("Arrays generated for eincore..")

start = time.time()
d = np.einsum(subscripts, a, b)
print("Numpy entry: ", d.flatten()[0])
finish = time.time()
Numpy_time = finish - start
print("Numpy time: ", Numpy_time)
subscripts= subscripts.replace("a","ä")
print(subscripts)

start = time.time()
d = einsum(subscripts, a, b)
print("Eincore entry: ", d.flatten()[0])
finish = time.time()
Eincore_time = finish - start
print("Eincore time: ", Eincore_time)
print("Speedup: ", Numpy_time / Eincore_time)
