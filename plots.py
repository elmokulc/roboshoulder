import json 
import os 
import utils
import plot_toolbox.ce_plotly as ce_plotly
import numpy as np 
import pandas as pd


landmark_dir = "./Landmarks/"
composite_data_dir ="./data_composites/"
stl_dir = "./STL/"
raw_data_dir ="./data_raw/"


with open("./data_proceed_full.json", "r") as f:
    data_bucket = json.load(f, object_hook=utils.deconvert)

for ii in range(len(os.listdir(composite_data_dir))):   
        data_file = sorted(os.listdir(composite_data_dir))[ii] 
        # pbar.set_description(f"Current file: {data_file}")       
        ID, side, bone, datatype  = tuple(data_file[:-4].split("_"))
        
        data_composite_points = pd.read_csv(composite_data_dir + data_file,  header=[0, 1])        
        points_compo = {"ref_point":{},
                    "full_point":{}}

        # Plot area bone scan 
        for j in range(3):
            area_name = f"area_{j}"
            point_full_compo = data_composite_points[area_name].astype(np.float64).values
            point_full_compo = point_full_compo[~np.isnan(point_full_compo)].reshape(-1, 3)   
            points_compo["full_point"][area_name] = point_full_compo
        

        
        R_compo2stl, T_compo2stl = (data_bucket[f"{ID}_{side}_{bone}"]["RT_compo2stl_full"]["R"],
                data_bucket[f"{ID}_{side}_{bone}"]["RT_compo2stl_full"]["T"])


        fig = ce_plotly.create_mesh3D(stl_file=f"{stl_dir}{ID}_{side}_Mesh_{bone}_cluster.stl",
                                        title=f"{ID}_{side}_{bone}",)

        points_scanned_compo = np.concatenate([points for key, points in points_compo["full_point"].items()], axis=0)
        
        color_list = ["#ff00ff", "#66ff66", "#0099ff"]
        for i, val in enumerate(points_compo["full_point"].items()):
                key, points_compo = val 
                points_stl = utils.apply_RT(points_compo, R_compo2stl, T_compo2stl)
        

                fig = ce_plotly.add_points(fig=fig,
                                        points=points_stl*1e3,
                                        name=f"{key}",
                                        # line=dict(color=color_list[i]), 
                                        marker_size=3,)


        # Plot cartilage scan 
        operator_columns = sorted(set([c for c, sub in data_composite_points.columns if "operator" in c]))
        for c in operator_columns:
                points_stl = utils.apply_RT(data_composite_points[c].values.astype(np.float64).reshape(-1,3), R_compo2stl, T_compo2stl)

                fig = ce_plotly.add_points(fig=fig,
                                points=points_stl*1e3,
                                name=f"{c}",
                                marker_size=3,)


        # Plot landmark 
        land_file = f"{ID}_{side}_{bone}_Landmarks.fcsv"
        data_ref_points = pd.read_csv(landmark_dir + land_file, skiprows=2)
        data_ref_points.x *= 1e-3
        data_ref_points.y *= 1e-3
        data_ref_points.z *= 1e-3

        for key, df in sorted(data_ref_points.groupby("label")):
            point_landmark = np.array([df.x.values, df.y.values, df.z.values]).astype(np.float64) 
            fig = ce_plotly.add_points(fig=fig,
                                points=np.ones_like(np.eye(3))*point_landmark.T *1e3,
                                name=f"Landmark_{key}",
                                marker_size=3,)

        # Plot composite reference points 
        ref_point_cols = sorted(set([c for c, sub in data_composite_points.columns if len(c)==3]))
        for c in ref_point_cols:
            points_stl = utils.apply_RT(data_composite_points[c].values.astype(np.float64).reshape(-1,3), R_compo2stl, T_compo2stl)

            fig = ce_plotly.add_points(fig=fig,
                            points=points_stl*1e3,
                            name=f"{c}",
                            marker_size=3,)
    
        
        fig.update_layout(showlegend=True,
                        width=1800,
                        height=1000,)
        
        fig.write_html(f"{utils.checkdir('./Figures/')}/{ID}_{side}_{bone}.html")