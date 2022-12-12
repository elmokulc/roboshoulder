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
import json
import plotly.graph_objects as go

def thrd(f):
    ''' This decorator executes a function in a Thread'''
    @wraps(f)
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
        
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

            point_scan_stl = apply_RT(
                point_scan_compo, Rcompo2stl, Tcompo2stl)
            res.append(abs(point_scan_stl - point_selected_stl))

        return np.concatenate(res, axis=0).flatten()-pen_pin_radius

def get_initial_tranform(tri_ref_points_compo, tri_ref_points_stl):
        
    X0 = np.zeros(6)
    return least_squares(cost_init, X0, method="lm", ftol=1e-12, xtol=1e-12, gtol=1e-10, args=(tri_ref_points_compo, tri_ref_points_stl,))

def cost_full(X, p3d, mesh):
    Rcompo2stl, Tcompo2stl = unpack(X)
    p_stl = apply_RT(p3d, Rcompo2stl, Tcompo2stl)
    dist = trimesh.proximity.signed_distance(mesh, p_stl)
    pen_pin_radius = 1e-3
    res = dist + pen_pin_radius

    print("X̅ res (mm) : {0}  \t σ (mm) : {1}".format(
        (np.sqrt(sum(res**2)) / len(res)) * 1e3, res.std() * 1e3), end="\r")

    return res

def get_full_transform(X0, points_scan_compo, bone_mesh):
    
    points_compo = np.concatenate([points for key, points in points_scan_compo.items()], axis=0)

    return least_squares(
            cost_full, X0, method="lm", ftol=1e-12, xtol=1e-12, gtol=1e-10,
            args=(points_compo, bone_mesh,))
    
    
    
table_stl = {
        "RS001":"3220",
        "RS002":"2320",
        "RS003":"2920",
        "RS004":"0820",
        "RS005":"2620"
        }

landmark_dir = "./Landmarks/"
composite_data_dir ="./data_composites/"
stl_dir = "./STL/"
raw_data_dir ="./data_raw/"
segmented = "./STL/segmented/"

data_local = {"RT_compo2stl_init": {},
                  "RT_compo2stl_full":{},
                  "data_stl":{},
                  "data_compo":{},
                  }  

data_file ='RS001_R_Scapula_composite.csv'
ID, side, bone, datatype  = tuple(data_file[:-4].split("_"))
land_file = f"{ID}_{side}_{bone}_Landmarks.fcsv"


data_composite_points = pd.read_csv(composite_data_dir + data_file,  header=[0, 1])
data_ref_points = pd.read_csv(landmark_dir + land_file, skiprows=2)
data_ref_points.x *= 1e-3
data_ref_points.y *= 1e-3
data_ref_points.z *= 1e-3

points_compo = {"ref_point":{},
                "full_point":{}}
points_stl = {"ref_point":{}}
    
for key, df in sorted(data_ref_points.groupby("label")):       
    key = key[1:]
                    
    if key in data_composite_points:
        point_ref_compo = data_composite_points[key].values.astype(np.float64)
        point_ref_compo = point_ref_compo[~np.isnan(point_ref_compo)].reshape(-1, 3)                        
        points_compo["ref_point"][key] = point_ref_compo.astype(np.float64)
        
        points_stl["ref_point"][key] = np.array([df.x.values, df.y.values, df.z.values]).astype(np.float64)                
        
        data_local["data_stl"].update({key:points_stl["ref_point"][key]})
        data_local["data_compo"].update({key:points_compo["ref_point"][key]})
        
sol = get_initial_tranform(tri_ref_points_compo = points_compo["ref_point"], tri_ref_points_stl = points_stl["ref_point"])
        # print(sol.optimality)
R_compo2stl, T_compo2stl = unpack(sol.x)
data_local["RT_compo2stl_init"]={"R": R_compo2stl, "T": T_compo2stl}

data_bucket = {}

def get_mesh(area, return_name=False):
    mesh = trimesh.exchange.load.load_mesh(f"{segmented}RS001_R_Mesh_Scapula_{area}.stl", 
                                                    type="stl")
    mesh.apply_scale(1e-3)
    if return_name:
        return mesh, f"{segmented}RS001_R_Mesh_Scapula_{area}.stl"
    else:
        return mesh


def cost_full_bis(X, p3d, mesh):
    Rcompo2stl, Tcompo2stl = unpack(X)
    p_stl = apply_RT(p3d, Rcompo2stl, Tcompo2stl)
    dist = trimesh.proximity.signed_distance(mesh, p_stl)
    pen_pin_radius = 1e-3
    res = dist + pen_pin_radius

    print("X̅ res (mm) : {0}  \t σ (mm) : {1}".format(
        (np.sqrt(sum(res**2)) / len(res)) * 1e3, res.std() * 1e3), end="\r")

    return res


def get_full_transform_bis(X, points0, points1, points2, mesh0, mesh1, mesh2):
    
    Rcompo2stl, Tcompo2stl = unpack(X)

    p_stl = apply_RT(points0, Rcompo2stl, Tcompo2stl)
    dist = trimesh.proximity.signed_distance(mesh0, p_stl)
    pen_pin_radius = 1e-3
    res0 = dist + pen_pin_radius
    
    p_stl = apply_RT(points1, Rcompo2stl, Tcompo2stl)
    dist = trimesh.proximity.signed_distance(mesh1, p_stl)
    pen_pin_radius = 1e-3
    res1 = dist + pen_pin_radius
    
    p_stl = apply_RT(points2, Rcompo2stl, Tcompo2stl)
    dist = trimesh.proximity.signed_distance(mesh2, p_stl)
    pen_pin_radius = 1e-3
    res2 = dist + pen_pin_radius
    
    res = np.concatenate((res0, res1, res2))
    
    return res.flatten()


def unpack_data(data_bucket):

    return data_bucket["area_0"]["points_compo"], data_bucket["area_1"]["points_compo"], data_bucket["area_2"]["points_compo"],  data_bucket["area_0"]["mesh"], data_bucket["area_1"]["mesh"], data_bucket["area_2"]["mesh"]
        
    

for j in range(3):
    area_name = f"area_{j}"
    # point_full_compo = data_composite_points[area_name].astype(np.float64).values
    # point_full_compo = point_full_compo[~np.isnan(point_full_compo)].reshape(-1, 3)   
    # points_compo["full_point"][area_name] = point_full_compo
    
    data_bucket[area_name] = {}
    data_bucket[area_name]["mesh"] = get_mesh(area_name)
    area_points = data_composite_points[area_name].astype(np.float64).values
    area_points = area_points[~np.isnan(area_points)].reshape(-1, 3)  
    data_bucket[area_name]["points_compo"] = area_points
    

    
X0 = pack(data_local["RT_compo2stl_init"]["R"],
        data_local["RT_compo2stl_init"]["T"])   

points0, points1, points2, mesh0, mesh1, mesh2 = unpack_data(data_bucket)
sol = least_squares(
            get_full_transform_bis, X0, method="lm", ftol=1e-12, xtol=1e-12, gtol=1e-10,
            args=(points0, points1, points2, mesh0, mesh1, mesh2,))



def RT_points(points, R, T): 
    return apply_RT(points, R, T)

R_sol , T_sol = unpack(sol.x)
# points0_stl = RT_points(points0, R_sol, T_sol)
# points1_stl = RT_points(points1, R_sol, T_sol)
# points2_stl = RT_points(points2, R_sol, T_sol)

for key, value in data_bucket.items():
    value["points_stl"] = RT_points(value["points_compo"], R_sol, T_sol)


# fig = ce_plotly.create_mesh3D(stl_file=f"{stl_dir}{ID}_{side}_Mesh_{bone}_cluster.stl",
#                                         title=f"{ID}_{side}_{bone}",)


fig = go.Figure()

for key, value in data_bucket.items():
    
    _, fname = get_mesh(key, return_name=True)
    
    _, mesh3D = ce_plotly.create_mesh3D(stl_file=fname, return_mesh=True)
    
    fig.add_trace(mesh3D)
    
    value["points_stl"] = RT_points(value["points_compo"], R_sol, T_sol)
    
    fig = ce_plotly.add_points(fig=fig,
                                        points=value["points_stl"]*1e3,
                                        name=f"{key}",
                                        # line=dict(color=color_list[i]), 
                                        marker_size=3,)

fig.update_layout(showlegend=True)
        
fig.write_html(f"{checkdir('./Figures/')}/{ID}_{side}_{bone}_segmented.html")