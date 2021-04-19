from .hptt import transpose as hptt_transpose
from .tblis import contract as tblis_contract
import numpy as np


def transpose(a, axes=None):
    at = hptt_transpose(a, axes=axes)
    return at


def tensordot(x, y, axes=2):
    return _tensordot(x, y, axes=axes)
    # return np.tensordot(x,y,axes=axes)


def einsum(*args, **kwargs):
    return _eincore(*args, **kwargs)


def _tensordot(a, b, axes=2):
    a_string = "".join([chr(i + 65) for i in range(0, a.ndim)])
    b_string = "".join([chr(i + a.ndim + 65) for i in range(0, b.ndim)])
    if isinstance(axes, int):
        if axes == 0:
            c_string = a_string + b_string
        else:
            b_string = a_string[-axes:] + b_string[axes:]
            c_string = symmetric_diff(a_string, b_string)
    elif isinstance(axes, tuple):
        #Not so confident about this...
        for i in range(len(axes[0])):
            b_string = b_string.replace(b_string[axes[1][i]], a_string[axes[0][i]])
        c_string = symmetric_diff(a_string, b_string)
    else:
        raise ValueError("Invalid axes type, must be int or tuple of lists/tuples")
    einsum_str = a_string + "," + b_string + "->" + c_string
    return _eincore(einsum_str, a, b)


def _eincore(*args, **kwargs):
    if len(args) == 3:
        subscripts, a, b = args[:3]
        ab_subs, out_subs = subscripts.split("->")
        a_subs, b_subs = ab_subs.split(",")
        a_sublist, b_sublist, out_sublist = list(a_subs), list(b_subs), list(out_subs)

        for subs in a_sublist, b_sublist, out_sublist:
            if len(subs) != len(set(subs)):
                raise NotImplementedError("Repeated subscripts not implemented")
        # Sum stragglers
        a, a_sublist = _sum_unique_axes(a, a_sublist, b_sublist, out_sublist)
        b, b_sublist = _sum_unique_axes(b, b_sublist, a_sublist, out_sublist)
        #  Handle ij,k->ij case
        if a.ndim == 0:
            return _transpose(a * b, b_sublist, out_sublist)
        if b.ndim == 0:
            return _transpose(b * a, a_sublist, out_sublist)

        a_subs, b_subs, out_subs = map(set, (a_sublist, b_sublist, out_sublist))
        if out_subs - (a_subs | b_subs):
            raise ValueError(
                "Output subscripts must be contained within input subscripts"
            )

        einsum_string = (
            "".join(a_sublist) + "," + "".join(b_sublist) + "->" + "".join(out_sublist)
        )

        return tblis_contract(einsum_string, a, b)
    else:
        # ij->ji or ij->
        subscripts, a = args[:2]
        a_subs, out_subs = subscripts.split("->")
        out_sublist = list(out_subs)
        a, a_sublist = _sum_unique_axes(a, list(a_subs), out_sublist)
        return _transpose(a, a_sublist, out_sublist)


def _transpose(in_arr, in_sublist, out_sublist):
    if set(in_sublist) != set(out_sublist):
        raise ValueError("Input and output subscripts don't match")
    for sublist in (in_sublist, out_sublist):
        if len(set(sublist)) != len(sublist):
            raise NotImplementedError("Repeated subscripts not implemented")

    in_idxs = {k: v for v, k in enumerate(in_sublist)}
    id_axes = [i for i in range(len(out_sublist))]
    axes = [in_idxs[s] for s in out_sublist]

    if len(axes) == 0 or (str(id_axes) == str(axes)):
        return in_arr

    return hptt_transpose(in_arr, axes=axes)


def _sum_unique_axes(in_arr, in_sublist, *keep_subs):
    # assumes no repeated subscripts
    assert len(in_sublist) == len(set(in_sublist))
    out_sublist = []
    sum_axes = []
    keep_subs = set([s for ks in keep_subs for s in ks])
    for idx, sub in enumerate(in_sublist):
        if sub in keep_subs:
            out_sublist.append(sub)
        else:
            sum_axes.append(idx)
    if sum_axes:
        return np.sum(in_arr, axis=tuple(sum_axes)), out_sublist
    else:
        return in_arr, out_sublist


def symmetric_diff(a_string, b_string):
    c_string = "".join([x for x in a_string if x not in set(b_string)])
    c_string += "".join([x for x in b_string if x not in set(a_string)])
    return c_string
