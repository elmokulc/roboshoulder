import os 
import numpy as np 
import cv2
from threading import Thread
from functools import wraps



def convert(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        return {"$array": x.tolist()}  # Make a tagged object
    raise TypeError(x)


def deconvert(x):
    if len(x) == 1:  # Might be a tagged object...
        key, value = next(iter(x.items()))  # Grab the tag and value
        if key == "$array":  # If the tag is correct,
            return np.array(value)  # cast back to array
    return x


def thrd(f):
    """This decorator executes a function in a Thread"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
        return thr

    return wrapper


def checkdir(directory="./outputs/"):
    """
    Creates dir not exists.

    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def compose_RT(R1, T1, R2, T2):
    """
    Composed 2 RT transformations
    """
    R, T = cv2.composeRT(R1, T1, R2, T2)[:2]
    return R.flatten(), T.flatten()


def invert_RT(R, T):
    """
    Inverts a RT transformation
    """
    Ti = -cv2.Rodrigues(-R)[0].dot(T)
    return -R, Ti


def apply_RT(P, R, T):
    """
    Applies RT transformation to 3D points P.
    """
    P = cv2.Rodrigues(R)[0].dot(P.T).T
    P += T.T
    return P
