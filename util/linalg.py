import numpy as np
from scipy.linalg import lapack, blas
from numpy.lib.stride_tricks import as_strided
import configparser
from scipy import linalg
config = configparser.ConfigParser()

try:
    from util import linalg_cython
    use_linalg_cython = True
except ImportError:
    use_linalg_cython = False


def _symmetrify_cython(A, upper=False):
    return linalg_cython.symmetrify(A, upper)


def _symmetrify_numpy(A, upper=False):
    triu = np.triu_indices_from(A,k=1)
    if upper:
        A.T[triu] = A[triu]
    else:
        A[triu] = A.T[triu]


def symmetrify(A, upper=False):
    """
    Take the square matrix A and make it symmetrical by copting elements from
    the lower half to the upper

    works IN PLACE.

    note: tries to use cython, falls back to a slower numpy version
    """
    if use_linalg_cython:
        _symmetrify_cython(A, upper)
    else:
        _symmetrify_numpy(A, upper)


def view(A, offset=0):
    from numpy.lib.stride_tricks import as_strided
    assert A.ndim == 2, "only implemented for 2 dimensions"
    assert A.shape[0] == A.shape[1], "attempting to get the view of non-square matrix?!"
    if offset > 0:
        return as_strided(A[0, offset:], shape=(A.shape[0] - offset, ), strides=((A.shape[0]+1)*A.itemsize, ))
    elif offset < 0:
        return as_strided(A[-offset:, 0], shape=(A.shape[0] + offset, ), strides=((A.shape[0]+1)*A.itemsize, ))
    else:
        return as_strided(A, shape=(A.shape[0], ), strides=((A.shape[0]+1)*A.itemsize, ))


def _diag_ufunc(A,b,offset,func):
    b = np.squeeze(b)
    assert b.ndim <= 1, "only implemented for one dimensional arrays"
    dA = view(A, offset); func(dA,b,dA)
    return A


def diag_add(A, b, offset=0):
    return _diag_ufunc(A, b, offset, np.add)


def diag_view(A, offset=0):
    assert A.ndim == 2, "only implemented for 2 dimensions"
    assert A.shape[0] == A.shape[1], "attempting to get the view of non-square matrix?!"
    if offset > 0:
        return as_strided(A[0, offset:], shape=(A.shape[0] - offset, ), strides=((A.shape[0]+1)*A.itemsize, ))
    elif offset < 0:
        return as_strided(A[-offset:, 0], shape=(A.shape[0] + offset, ), strides=((A.shape[0]+1)*A.itemsize, ))
    else:
        return as_strided(A, shape=(A.shape[0], ), strides=((A.shape[0]+1)*A.itemsize, ))


def tdot(mat, out=None):
    return tdot_blas(mat, out)


def tdot_blas(mat, out=None):
    """returns np.dot(mat, mat.T), but faster for large 2D arrays of doubles."""
    if (mat.dtype != 'float64') or (len(mat.shape) != 2):
        return np.dot(mat, mat.T)
    nn = mat.shape[0]
    if out is None:
        out = np.zeros((nn, nn))
    else:
        assert(out.dtype == 'float64')
        assert(out.shape == (nn, nn))
        # FIXME: should allow non-contiguous out, and copy output into it:
        assert(8 in out.strides)
        # zeroing needed because of dumb way I copy across triangular answer
        out[:] = 0.0

    # # Call to DSYRK from BLAS
    mat = np.asfortranarray(mat)
    out = blas.dsyrk(alpha=1.0, a=mat, beta=0.0, c=out, overwrite_c=1,
                     trans=0, lower=0)

    symmetrify(out, upper=True)
    return np.ascontiguousarray(out)


def jitchol(A, maxtries=5):
    A = np.ascontiguousarray(A)
    L, info = lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise linalg.LinAlgError("not pd: non-positive diagonal elements")
        jitter = diagA.mean() * 1e-6
        num_tries = 1
        while num_tries <= maxtries and np.isfinite(jitter):
            try:
                L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
                return L
            except:
                jitter *= 10
            finally:
                num_tries += 1
        raise linalg.LinAlgError("not positive definite, even with jitter.")
    import traceback
    try: raise
    except:
        logging.warning('\n'.join(['Added jitter of {:.10e}'.format(jitter),
            '  in '+traceback.format_list(traceback.extract_stack(limit=3)[-2:-1])[0][2:]]))
    return L


def dpotri(A, lower=1):
    A = force_F_ordered(A)
    R, info = lapack.dpotri(A, lower=lower) #needs to be zero here, seems to be a scipy bug

    symmetrify(R)
    return R, info


def dtrtri(L):
    L = force_F_ordered(L)
    return lapack.dtrtri(L, lower=1)[0]


def dtrtrs(A, B, lower=1, trans=0, unitdiag=0):
    A = np.asfortranarray(A)
    #Note: B does not seem to need to be F ordered!
    return lapack.dtrtrs(A, B, lower=lower, trans=trans, unitdiag=unitdiag)


def pdinv(A, *args):
    L = jitchol(A, *args)
    logdet = 2.*np.sum(np.log(np.diag(L)))
    Li = dtrtri(L)
    Ai, _ = dpotri(L, lower=1)
    symmetrify(Ai)
    return Ai, L, Li, logdet


def force_F_ordered(A):
    """
    return a F ordered version of A, assuming A is triangular
    """
    if A.flags['F_CONTIGUOUS']:
        return A
    print("why are your arrays not F order?")
    return np.asfortranarray(A)


def dpotrs(A, B, lower=1):
    """
    Wrapper for lapack dpotrs function
    :param A: Matrix A
    :param B: Matrix B
    :param lower: is matrix lower (true) or upper (false)
    :returns:
    """
    A = force_F_ordered(A)
    return lapack.dpotrs(A, B, lower=lower)