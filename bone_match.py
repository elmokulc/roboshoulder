from fileinput import FileInput
import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm
from scipy.optimize import least_squares
import trimesh
import cv2
import plot_toolbox.ce_plotly as ce_plotly
from threading import Thread
from functools import wraps
import time
import json

table_composite = {
    "humerus": {
        "0": "humerus_3220G",
        "1": "humerus_3220D",
        "2": "humerus_0820D",
        "3": "humerus_2320D",
        "4": "humerus_2320G",
        "5": "humerus_2920G",
        "6": "humerus_2920D",
        "7": "humerus_0820G",
        "8": "humerus_2620D",
        "9": "humerus_2620G",
    },
    "collarbone": {
        "0": "collarbone_2920G",
        "1": "collarbone_2920D ",
        "2": "collarbone_2620D",
        "3": "collarbone_2620G",
        "4": "collarbone_0820G",
        "5": "collarbone_0820D",
        "6": "collarbone_2320G",
        "7": "collarbone_2320D",
        "8": "collarbone_3220G",
        "9": "collarbone_3220D",
    },
    "scapula": {
        "0": "scapula_2920D",
        "1": "scapula_2320D",
        "2": "scapula_0820D",
        "3": "scapula_3220D",
        "4": "scapula_2620D",
        "5": "scapula_0820G",
        "6": "scapula_2620G",
        "7": "scapula_2920G",
        "8": "scapula_2320G",
        "9": "scapula_3220G",
    },
}

table_stl = {
    "RS001": "3220",
    "RS002": "2320",
    "RS003": "2920",
    "RS004": "0820",
    "RS005": "2620",
}


landmark_dir = "./Landmarks/"
composite_data_dir = "./data_composites/"
stl_dir = "./STL/"
raw_data_dir = "./data_raw/"


for raw_file in os.listdir(raw_data_dir):
    _, bonetype, ID_side = tuple(raw_file[:-4].split("_"))
    # print(bonetype, ID_side)
    ID = ID_side[:-1]
    side = ID_side[-1]

    if side == "G":
        side = "L"
    elif side == "D":
        side = "R"
    if bonetype == "collarbone":
        bonetype = "clavicle"

    ID = [key for key, value in table_stl.items() if ID == value][0]

    filename = f"{ID}_{side}_{bonetype.capitalize()}_composite.csv"

    shutil.copy(raw_data_dir + raw_file, composite_data_dir + filename)

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


def unpack(X):
    """_summary_

    Args:
        X (array): shape (1,6) or (6,)

    Returns:
        array : shape (2,3)
    """
    return X.reshape(2, 3)


def pack(R, T):
    """_summary_

    Args:
        R (array): shape (1,3) or (3,)
        T (array): shape (1,3) or (3,)

    Returns:
        array : shape (1,6) or (6,)
    """

    return np.concatenate((R, T))


def cost_init(X, ref_points_compo, ref_points_stl):
    Rcompo2stl, Tcompo2stl = unpack(X)
    pen_pin_radius = 1e-3
    res = []
    for key in ref_points_compo.keys():
        point_scan_compo = ref_points_compo[key].reshape(-1, 3)
        point_selected_stl = ref_points_stl[key].reshape(-1, 3)

        point_scan_stl = apply_RT(point_scan_compo, Rcompo2stl, Tcompo2stl)
        res.append(abs(point_scan_stl - point_selected_stl))

    return np.concatenate(res, axis=0).flatten() - pen_pin_radius


def get_initial_tranform(tri_ref_points_compo, tri_ref_points_stl):

    X0 = np.zeros(6)
    return least_squares(
        cost_init,
        X0,
        method="lm",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-10,
        args=(
            tri_ref_points_compo,
            tri_ref_points_stl,
        ),
    )


def cost_full(X, p3d, mesh):
    Rcompo2stl, Tcompo2stl = unpack(X)
    p_stl = apply_RT(p3d, Rcompo2stl, Tcompo2stl)
    dist = trimesh.proximity.signed_distance(mesh, p_stl)
    pen_pin_radius = 1e-3
    res = dist + pen_pin_radius

    # print("X̅ res (mm) : {0}  \t σ (mm) : {1}".format(
    #     (np.sqrt(sum(res**2)) / len(res)) * 1e3, res.std() * 1e3), end="\r")

    return res


def get_full_transform(X0, points_scan_compo, bone_mesh):

    points_compo = np.concatenate(
        [points for key, points in points_scan_compo.items()], axis=0
    )

    return least_squares(
        cost_full,
        X0,
        method="lm",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-10,
        args=(
            points_compo,
            bone_mesh,
        ),
    )


@thrd
def computation(data_file, Finished, index, size):
    data_local = {"RT_compo2stl_init": {}, "RT_compo2stl_full": {}}

    ID, side, bone, datatype = tuple(data_file[:-4].split("_"))
    land_file = f"{ID}_{side}_{bone}_Landmarks.fcsv"

    data_composite_points = pd.read_csv(composite_data_dir + data_file, header=[0, 1])
    data_ref_points = pd.read_csv(landmark_dir + land_file, skiprows=2)
    data_ref_points.x *= 1e-3
    data_ref_points.y *= 1e-3
    data_ref_points.z *= 1e-3

    points_compo = {"ref_point": {}, "full_point": {}}
    points_stl = {"ref_point": {}}

    for key, df in sorted(data_ref_points.groupby("label")):
        key = key[1:]

        if key in data_composite_points:
            point_ref_compo = data_composite_points[key].values.astype(np.float64)
            point_ref_compo = point_ref_compo[~np.isnan(point_ref_compo)].reshape(-1, 3)
            points_compo["ref_point"][key] = point_ref_compo.astype(np.float64)

            points_stl["ref_point"][key] = np.array(
                [df.x.values, df.y.values, df.z.values]
            ).astype(np.float64)
            data_local["data_stl"] = points_stl["ref_point"][key]
            data_local["data_compo"] = points_stl["ref_point"][key]
    try:
        sol = get_initial_tranform(
            tri_ref_points_compo=points_compo["ref_point"],
            tri_ref_points_stl=points_stl["ref_point"],
        )
        Finished[index] = True
        # print(sol.optimality)
        R_compo2stl, T_compo2stl = unpack(sol.x)
        data_local["RT_compo2stl_init"] = {"R": R_compo2stl, "T": T_compo2stl}

        for j in range(3):
            area_name = f"area_{j}"
            point_full_compo = (
                data_composite_points[area_name].astype(np.float64).values
            )
            point_full_compo = point_full_compo[~np.isnan(point_full_compo)].reshape(
                -1, 3
            )
            points_compo["full_point"][area_name] = point_full_compo

        X_init = pack(
            data_local["RT_compo2stl_init"]["R"], data_local["RT_compo2stl_init"]["T"]
        )

        bone_mesh = trimesh.exchange.load.load_mesh(
            f"{stl_dir}{ID}_{side}_Mesh_{bone}_cluster.stl", type="stl"
        )
        bone_mesh.apply_scale(1e-3)

        sol_full = get_full_transform(
            X0=X_init, points_scan_compo=points_compo["full_point"], bone_mesh=bone_mesh
        )
        R_compo2stl, T_compo2stl = unpack(sol_full.x)
        data_local["RT_compo2stl_full"] = {"R": R_compo2stl, "T": T_compo2stl}

    except TypeError as e:
        print("==========================================================")
        print(f"Fail: {ID}_{side}_{bone}")
        print(e)
        print("==========================================================")
        data_local = {
            "RT_compo2stl_init": {"R": None, "T": None},
            "RT_compo2stl_full": {"R": None, "T": None},
        }

        pass

    Finished[index + int(size / 2)] = True
    data_local
    json_string = ",\n".join(json.dumps(data_bucket, indent=4, default=convert).split(", "))
    
    with open(f"{checkdir('./data_processed/')}/{ID}_{side}_{bone}_data.json", "w") as f:
        f.write(json_string)



t_bucket = []
size = 2 * len(os.listdir(composite_data_dir))
Finished = np.zeros(size, dtype=bool)

t0 = time.time()

for i, data_file in enumerate(sorted(os.listdir(composite_data_dir))):
    ID, side, bone, datatype = tuple(data_file[:-4].split("_"))
    t_bucket.append(
        computation(
            data_file=data_file,
            Finished=Finished,
            index=i,
            size=size,
        )
    )

with tqdm(total=size, colour="red") as pbar:
    progress = 0
    while progress < size:
        progress = int(np.sum(Finished))
        time.sleep(0.2)
        old_value = pbar.n
        new_value = progress - old_value
        if new_value != old_value:
            pbar.update(new_value)

for t in t_bucket:
    t.join()

print(f"elapse time= {time.time() - t0} s")

data_bucket = {}

for i, data_file in enumerate(sorted(os.listdir(composite_data_dir))):
    ID, side, bone, datatype = tuple(data_file[:-4].split("_"))   
    data_bucket.update({f"{ID}_{side}_{bone}":{}}) 
    with open(f"{checkdir('./data_processed/')}/{ID}_{side}_{bone}_data.json") as f:
        data_bucket[f"{ID}_{side}_{bone}"] = json.load(f, object_hook=deconvert)

json_string = ",\n".join(json.dumps(data_bucket, indent=4, default=convert).split(", "))
with open("./data.json", "w") as f:
    f.write(json_string)
