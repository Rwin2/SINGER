import os
from re import I
import numpy as np
import torch
from torchvision.io import write_video
from scipy.signal import butter,lfilter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import figs.utilities.trajectory_helper as th
from typing import List, Tuple, Union
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from scipy.spatial.transform import Rotation as R

def pos2vel(T,P):
    """
    Compute the velocity from the position data.
    """
    V = np.zeros_like(P)

    for i in range(1,len(T)):
        V[:,i] = (P[:,i]-P[:,i-1])/(T[i]-T[i-1])
    V[:,0] = V[:,1]

    return V

def butter_lowpass_filter(data):
    """
    Butterworth low-pass filter.
    """
    b, a = butter(5, 3.0, fs=50.0)
    y = lfilter(b, a, data)
    return y

def data_check(data:dict):
    # Keys reference
    new_keys = ['Imgs', 'Tact', 'Uact', 'Xref', 'Uref', 'Xest', 'Xext', 'Adv', 'Tsol', 'obj', 'n_im']
    old_keys = ['Tact', 'Xref', 'Uref', 'Xact', 'Uact', 'Adv', 'Tsol', 'Imgs', 'tXds', 'n_im']
    idk_keys = ['Tact', 'Xref', 'Uref', 'Xact', 'Uact', 'Adv', 'Tsol', 'Imgs', 'n_im']

    # Load the flight data
    data_keys = list(data.keys())
    if data_keys == new_keys:
        pass
    elif data_keys == old_keys:
        data = {
            'Imgs': data['Imgs'],
            'Tact': data['Tact'],
            'Uact': data['Uact'],
            'Xref': data['Xref'],
            'Uref': data['Uref'],
            'Xest': data['Xact'],
            'Xext': data['Xact'],
            'Adv': data['Adv'],
            'Tsol': data['Tsol'],
            'obj': th.tXU_to_obj(data['tXds']),
            'n_im': data['n_im']
        }
    elif data_keys == idk_keys:
        print('data_check: idk_keys. FIX ME! Missing objective. Maybe not important.')
        data = {
            'Imgs': data['Imgs'],
            'Tact': data['Tact'],
            'Uact': data['Uact'],
            'Xref': data['Xref'],
            'Uref': data['Uref'],
            'Xest': data['Xact'],
            'Xext': data['Xact'],
            'Adv': data['Adv'],
            'Tsol': data['Tsol'],
            'obj': None,
            'n_im': data['n_im']
        }
    else:
        print("Data keys do not match expected keys")
    
    # Check for NaN values in the data
    if np.isnan(data['Xext'][3:6,:]).any():
        for i in range(1,data['Xext'].shape[1]):
            data['Xext'][3:6,i] = (data['Xext'][0:3,i]-data['Xext'][0:3,i-1])/(data['Tact'][i]-data['Tact'][i-1])
        data['Xext'][3:6,0] = data['Xext'][3:6,1]
    
    return data

def preprocess_trajectory_data(folder:str,Nfiles:Union[None,int]=None,
                                dt_trim:float=0.0,land_check:bool=True,z_floor:float=0.0):
    # ============================================================================
    # Unpack the data
    # ============================================================================
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path = os.path.join(workspace_path,"cohorts",folder)
    folder_path = os.path.join(cohort_path,'flight_data')
    data_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]
    data_files.sort()

    # Trim file count if need be
    if Nfiles is not None:
        data_files = data_files[-Nfiles:]
    
    print("==============================||   Flight Data Loader   ||==============================||")
    print("File Name :",folder)
    print("File Count:",len(data_files))
    print("Trim Time :",dt_trim)
    print("Land Check:",land_check)

    flights = []
    for file in data_files:
        # Load Data
        raw_data = torch.load(file,weights_only=False)

        # Check if the data is using the old format
        raw_data = data_check(raw_data)
        
        # Land check and pad accordingly
        if land_check:
            if np.any(raw_data['Xest'][2,:] > z_floor):
                idx = np.argmax(raw_data['Xest'][2, :] > z_floor)

                raw_data['Xest'][:,idx:] = raw_data['Xest'][:,idx].reshape(-1,1)
                raw_data['Xest'][3:6,idx:] = 0.0
                raw_data['Uact'][:,idx:] = 0.0
            
        # Trim the data
        if dt_trim > 0.0:
            idx = np.argmax(raw_data['Tact'] > raw_data['Tact'][-1]-dt_trim)
        else:
            idx = raw_data['Tact'].shape[0]

        # Create flight data
        data = {}
        data['Tref'] = raw_data['Tact'][:idx+1]
        data['Xref'] = raw_data['Xref'][:,:idx+1]
        # # Print Xref values where Tref is nonzero
        # nonzero_tref_mask = raw_data['Tact'] != 0
        # xref_nonzero = raw_data['Xref'][:, nonzero_tref_mask]
        # print("Xref values for Tref nonzero:\n", xref_nonzero)
        data['Tact'] = raw_data['Tact'][:idx+1]
        data['Xact'] = raw_data['Xest'][:,:idx+1]
        data['Uact'] = raw_data['Uact'][:,:idx+1]
        data['Tsol'] = raw_data['Tsol'][:,:idx+1]

        flights.append(data)
    
    return flights

def compute_TTE(Xact:np.ndarray,Xref:np.ndarray):
    """
    Compute the Trajectory Tracking Error (TTE) for the given data.
    """
    Ndata = Xact.shape[1]

    TTE = np.zeros(Ndata)
    for i in range(Ndata):
        TTE[i] = np.min(np.linalg.norm(Xact[0:3,i].reshape(-1,1)-Xref[0:3,:],axis=0))

    return TTE

def compute_PP(Xact:np.ndarray,Xref:np.ndarray,thresh:float=0.2):
    """
    Compute the Proximity Percentile (PP) for the given data.
    """
    Ndata = Xact.shape[1]

    count = 0
    for i in range(Ndata):
        pos_nearest = np.min(np.linalg.norm(Xact[0:3,i].reshape(-1,1)-Xref[0:3,:],axis=0))

        if pos_nearest < thresh:
            count += 1

    PP = 100*(count/Ndata)

    return PP

def compute_TDT(Xact:np.ndarray):
    """
    Compute the Total Distance Traveled (TDT) for the given data.
    """
    Ndata = Xact.shape[1]

    TDT = 0.0
    for i in range(1,Ndata):
        TDT += np.linalg.norm(Xact[0:3,i]-Xact[0:3,i-1])

    return TDT

def _active_mask_from_Uact(Uact: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Contiguous mask from first to last column where any input |u|>eps."""
    if Uact.ndim != 2 or Uact.shape[1] == 0:
        return np.zeros((0,), dtype=bool)
    nz_any = np.any(np.abs(Uact) > eps, axis=0)  # shape (K,)
    if not np.any(nz_any):
        return nz_any
    i0 = int(np.argmax(nz_any))
    i1 = len(nz_any) - int(np.argmax(nz_any[::-1]))  # one past last True
    m = np.zeros_like(nz_any, dtype=bool)
    m[i0:i1] = True
    return m

def _vec_flu_to_frd(S, v_xyz):
    v = np.asarray(v_xyz, dtype=float)
    return S @ v

def _quat_flu_to_frd_xyzw(q_xyzw):
    q = np.asarray(q_xyzw, dtype=float)
    q_frd = q.copy()
    q_frd[1] *= -1.0  # flip y
    q_frd[2] *= -1.0  # flip z
    # (optional) renormalize to be safe:
    q_frd /= np.linalg.norm(q_frd)
    return q_frd

def _first_valid_col_pos(X):
    # returns first column index where x,y,z are all finite; falls back to 0
    cols = np.where(np.all(np.isfinite(X[0:3, :]), axis=0))[0]
    return int(cols[0]) if cols.size else 0

def _detect_major_reversal(X, threshold_angle_deg=120):
    """
    Detects the first major reversal in trajectory direction.
    Returns the index before the reversal, or None if not found.
    Also prints which trajectory, where, and when the reversal happens.
    """
    pos = X[0:3, :]
    v = np.diff(pos, axis=1)
    v_norm = np.linalg.norm(v, axis=0)
    v_unit = np.zeros_like(v)
    mask = v_norm > 1e-6
    v_unit[:, mask] = v[:, mask] / v_norm[mask]

    # Compute angle between consecutive velocity vectors
    dot = np.sum(v_unit[:, :-1] * v_unit[:, 1:], axis=0)
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.arccos(dot) * 180.0 / np.pi

    # Find first index where angle exceeds threshold
    idx = np.argmax(angles > threshold_angle_deg)
    if angles[idx] > threshold_angle_deg:
        reversal_idx = idx + 1  # +1 to include up to this point
        reversal_pos = pos[:, reversal_idx]
        print(f"Major reversal detected at index {reversal_idx}, position {reversal_pos}, time (if available): {X[0, reversal_idx] if X.shape[0] > 0 else 'N/A'}")
        return reversal_idx
    return None

def plot_flight(cohort:str,
                # scenes:List[Tuple[str,str]],
                query_location,
                Nfiles:Union[None,int]=None,
                dt_trim:float=0.0,Nrnd:int=3,
                land_check:bool=False,
                plot_raw:bool=True,
                strip_idle:bool=False,
                baseline:bool=False,
                onboard:bool=False,
                ssv:bool=False,
                filter:bool=False,
                reset_z:bool=False
                ):
    
    flights = preprocess_trajectory_data(cohort,Nfiles,dt_trim,land_check=land_check)

    if strip_idle:
        for f in flights:
            m = _active_mask_from_Uact(f['Uact'])
            if m.size == 0 or not np.any(m):
                continue  # nothing to trim

            # 1D time arrays
            if 'Tact' in f: f['Tact'] = f['Tact'][m]
            if 'Tref' in f and f['Tref'] is not None: f['Tref'] = f['Tref'][m]

            # 2D (rows, time) arrays
            for k in ['Uact','Uref','Xact','Xref','Xest','Xext']:
                if k in f and f[k] is not None and f[k].ndim == 2 and f[k].shape[1] == m.shape[0]:
                    f[k] = f[k][:, m]

            # Tsol is (5, K) — mask along time
            if 'Tsol' in f and f['Tsol'] is not None and f['Tsol'].ndim == 2 and f['Tsol'].shape[1] == m.shape[0]:
                f['Tsol'] = f['Tsol'][:, m]

    # ============================================================================
    # Compute the effective control frequency
    # ============================================================================

    Tsols = []
    for flight in flights:
        Tsols.append(flight['Tsol'][4,:])
    print("Effective Control Frequency (Hz): ",[np.around(1/np.mean(Tsol),Nrnd) for Tsol in Tsols])

    # ============================================================================
    # Plot the 3D Trajectories
    # ============================================================================

    # Plot limits
    plim = np.array([
        [ -5.0,  5.0],
        [ -3.5,  3.5],
        [  0.0, -3.0]])
    
    xlim = plim[0,:]
    ylim = plim[1,:]
    zlim = plim[2,:]
    ratio = plim[:,1]-plim[:,0]

    # Check if most of the trajectory is at abs(z) > 1.8, and if so, move it to -1.0
    if reset_z:
        for flight in flights:
            Xact = flight.get('Xact', None)
            if Xact is not None and Xact.shape[1] > 0:
                z = Xact[2, :]
                if np.mean(np.abs(z) > 1.8) > 0.5:
                    Xact[2, :] = -1.0
                    flight['Xact'] = Xact
            Xref = flight.get('Xref', None)
            if Xref is not None and Xref.shape[1] > 0:
                z_ref = Xref[2, :]
                if np.mean(np.abs(z_ref) > 1.8) > 0.5:
                    Xref[2, :] = -1.0
                    flight['Xref'] = Xref

    if ssv and filter:
        # Detect trajectory reversals due to manual control.
        for f in flights:
            Xact = f.get('Xact', None)
            if Xact is None or Xact.shape[1] < 3:
                continue
            rev_idx = _detect_major_reversal(Xact)
            if rev_idx is not None and rev_idx < Xact.shape[1]:
                # Crop all time-dependent arrays
                for k in ['Tact', 'Tref']:
                    if k in f and f[k] is not None and f[k].ndim == 1 and f[k].shape[0] >= rev_idx:
                        f[k] = f[k][:rev_idx]
                for k in ['Xact', 'Xref', 'Xest', 'Xext', 'Uact', 'Uref']:
                    if k in f and f[k] is not None and f[k].ndim == 2 and f[k].shape[1] >= rev_idx:
                        f[k] = f[k][:, :rev_idx]
                if 'Tsol' in f and f['Tsol'] is not None and f['Tsol'].ndim == 2 and f['Tsol'].shape[1] >= rev_idx:
                    f['Tsol'] = f['Tsol'][:, :rev_idx]

            # Crop trajectory if z positions explode (e.g., abs(z) > 10)
            z = Xact[2, :]
            explode_thresh = 1.8
            if np.any(np.abs(z) > explode_thresh):
                idx_explode = np.argmax(np.abs(z) > explode_thresh)
                print(f"Z explosion detected at index {idx_explode}, z={z[idx_explode]}")
                # Crop all time-dependent arrays
                for k in ['Tact', 'Tref']:
                    if k in f and f[k] is not None and f[k].ndim == 1 and f[k].shape[0] >= idx_explode:
                        f[k] = f[k][:idx_explode]
                for k in ['Xact', 'Xref', 'Xest', 'Xext', 'Uact', 'Uref']:
                    if k in f and f[k] is not None and f[k].ndim == 2 and f[k].shape[1] >= idx_explode:
                        f[k] = f[k][:, :idx_explode]
                if 'Tsol' in f and f['Tsol'] is not None and f['Tsol'].ndim == 2 and f['Tsol'].shape[1] >= idx_explode:
                    f['Tsol'] = f['Tsol'][:, :idx_explode]
    
    # Get the relevant data from flights (trajectories and speed spectrum variables)
    XXact,Spd = [],[]
    for flight in flights:
        XXact.append(flight['Xact'])
        Spd.append(np.linalg.norm(flight['Xact'][3:6,:],axis=0))

    # Determine the colormap
    sdp_all = np.hstack(Spd)
    norm = plt.Normalize(sdp_all.min(), sdp_all.max())
    cmap = plt.get_cmap('viridis')

    # Get the reference trajectory
    # Xref = flights[0]['Xref']

    if onboard:
        S = np.diag([1, -1, -1])
        for f in flights:
            # Transform actual state if present
            if 'Xact' in f and f['Xact'] is not None:
                X = f['Xact']
                # X[0:3, :] = S @ X[0:3, :]
                # X[0:3, :] = X[0:3, :]
                X[1,: ] *= -1.0
                if X.shape[0] >= 10:  # if quaternion rows exist at 6..9
                    X[6:10, :] = np.apply_along_axis(_quat_flu_to_frd_xyzw, 0, X[6:10, :])
                f['Xact'] = X  # (in-place; assignment optional)

            # Transform reference state if present
            if 'Xref' in f and f['Xref'] is not None:
                Xr = f['Xref']
                Xr[0:3, :] = S @ Xr[0:3, :]
                if Xr.shape[0] >= 10:
                    Xr[6:10, :] = np.apply_along_axis(_quat_flu_to_frd_xyzw, 0, Xr[6:10, :])
                f['Xref'] = Xr  # (in-place; assignment optional)

            Xref = f.get('Xref', None)
            Xact = f.get('Xact', None)
            if Xref is None or Xact is None:
                continue

            jr = _first_valid_col_pos(Xref)
            ja = _first_valid_col_pos(Xact)

            # translation to put Xact start at Xref start
            t = Xref[0:3, jr] - Xact[0:3, ja]   # shape (3,)
            Xact[0:3, :] += t[:, None]          # shift all positions
            f['Xact'] = Xact  # (in-place; assignment optional)



        # # Align Xref start position to Xact start position (per flight)
        # Xref_aligned = []
        # for i, f in enumerate(flights):
        #     Xref_i = f.get('Xref', None)
        #     if Xref_i is None:
        #         Xref_aligned.append(None)
        #         continue

        #     Xref_i = Xref_i.copy()
        #     j_ref = _first_valid_col_pos(Xref_i)
        #     j_act = _first_valid_col_pos(f['Xact'])

        #     # translation to put Xref start at Xact start
        #     t = f['Xact'][0:3, j_act] - Xref_i[0:3, j_ref]   # shape (3,)
        #     Xref_i[0:3, :] += t[:, None]

        #     Xref_aligned.append(Xref_i)

        # # Preserve your original single-array access:
        # Xref = Xref_aligned[0] if Xref_aligned and Xref_aligned[0] is not None else None

    XXref = []
    for f in flights:
        if 'Xref' in f and f['Xref'] is not None:
            XXref.append(f['Xref'])
        else:
            XXref.append(None)
    
    # Initialize the figure
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_zlim(zlim)
    ax1.invert_yaxis()
    ax1.invert_zaxis()
    ax1.set_box_aspect(ratio)  # aspect ratio is 1:1:1 in data space

    if not ssv:
        # Plot the reference trajectory
        ax1.plot(Xref[0,:],Xref[1,:],Xref[2,:],label='Desired',color='#4d4d4d',linestyle='--')
    # else:
    #     # ax1.plot(Xref[0,:],Xref[1,:],Xref[2,:],label='Actual',color='#4d4d4d',linestyle='--')
    #     for Xref in XXref:
    #         if Xref is None:
    #             continue
    #         ax1.plot(Xref[0, :], Xref[1, :], -Xref[2, :],
    #                 linestyle='--', color='#3b528b', linewidth=1.0, label="Reference")

    # Extract and plot the speed lines
    if not ssv:
        for Xact,spd in zip(XXact,Spd):
            xyz = Xact[0:3,:].T
            segments = np.stack([xyz[:-1,:], xyz[1:,:]], axis=1)
            label = 'Xact' if idx == len(XXact) - 1 else None
            lc = Line3DCollection(segments, alpha=0.5,linewidths=2, colors=cmap(norm(spd)), label=label)    #     segments = np.concatenate([points[:-1], points[1:]], axis=1).T
            ax1.add_collection(lc)
    else:
        if onboard:
            for idx, (Xref, spd) in enumerate(zip(XXref, Spd)):
                xyz = Xref[0:3,:].T
                segments = np.stack([xyz[:-1,:], xyz[1:,:]], axis=1)
                label = 'Xref' if idx == len(XXref) - 1 else None
                lc = Line3DCollection(segments, alpha=0.5,linewidths=2, colors=cmap(norm(spd)), label=label)    #     segments = np.concatenate([points[:-1], points[1:]], axis=1).T
                ax1.add_collection(lc)
        else:
            for idx, (Xact, spd) in enumerate(zip(XXact, Spd)):
                xyz = Xact[0:3, :].T
                segments = np.stack([xyz[:-1, :], xyz[1:, :]], axis=1)
                # Only label the last trajectory
                label = 'Xact' if idx == len(XXact) - 1 else None
                lc = Line3DCollection(segments, alpha=0.5, linewidths=2, colors=cmap(norm(spd)), label=label)
                ax1.add_collection(lc)

    if not ssv:
        ax1.plot([], [], [], label='Actual',color='#3b528b', alpha=0.6)
    # else:
        # if onboard:
            # ax1.plot([], [], [], label='Actual',color='#3b528b', alpha=0.6)

    if ssv and query_location is not None:
        # Plot the query location as a dot
        ax1.scatter(query_location[0], query_location[1], query_location[2], color='red', s=60, label='Query Location', zorder=10)

        # Plot a circle of radius 1.5m around the query location (in the XY plane)
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = (query_location[0] + 1.5 * np.cos(theta))
        circle_y = (query_location[1] + 1.5 * np.sin(theta))
        circle_z = np.full_like(theta, query_location[2])
        ax1.plot(circle_x, circle_y, circle_z, color='red', linestyle=':', linewidth=2, label='1.5m Radius')

        # Plot the quad frame at the first location in the trajectory
        if onboard:
            if len(XXref) > 0 and XXref[0].shape[1] > 0:
                quad_frame(XXact[0][:, 0], ax1, scale=1.0)
                # Compute and print the normalized distance to the boundary of the 1.5m circle for all trajectories
                for idx, Xref in enumerate(XXref):
                    if Xref.shape[1] == 0:
                        continue
                    goal_pos = np.array(query_location[:2])
                    final_pos_xy = Xref[0:2, -1]
                    dist_to_center = np.linalg.norm(final_pos_xy - goal_pos)
                    dist_initial = np.linalg.norm(Xref[0:2, 0] - goal_pos - 1.5)
                    dist_to_boundary = abs(dist_to_center - 1.5)
                    # Normalize by the trajectory length in the XY plane
                    traj_xy = Xref[0:2, :]
                    traj_len = np.sum(np.linalg.norm(traj_xy[:, 1:] - traj_xy[:, :-1], axis=0))
                    normalized_dist = dist_to_boundary / dist_initial if dist_initial > 0 else np.nan
                    # print(f"Trajectory {idx}: Normalized distance from 1.5m boundary as progress = {normalized_dist:.3f}")
                    print(f"Trajectory {idx}:  Normalized distance from 1.5m boundary = {normalized_dist:.3f}")
        else:
            if len(XXact) > 0 and XXact[0].shape[1] > 0:
                quad_frame(XXact[0][:, 0], ax1, scale=1.0)
                # Compute and print the normalized distance to the boundary of the 1.5m circle for all trajectories
                for idx, Xact in enumerate(XXact):
                    if Xact.shape[1] == 0:
                        continue
                    goal_pos = np.array(query_location[:2])
                    final_pos_xy = Xact[0:2, -1]
                    dist_to_center = np.linalg.norm(final_pos_xy - goal_pos)
                    dist_initial = np.linalg.norm(Xact[0:2, 0] - goal_pos - 1.5)
                    dist_to_boundary = abs(dist_to_center - 1.5)
                    # Normalize by the trajectory length in the XY plane
                    traj_xy = Xact[0:2, :]
                    traj_len = np.sum(np.linalg.norm(traj_xy[:, 1:] - traj_xy[:, :-1], axis=0))
                    normalized_dist = dist_to_boundary / dist_initial if dist_initial > 0 else np.nan
                    # print(f"Trajectory {idx}: Normalized distance from 1.5m boundary as progress = {normalized_dist:.3f}")
                    print(f"Trajectory {idx}:  Normalized distance from 1.5m boundary = {normalized_dist:.3f}")

    # Add the colorbar
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax1,orientation='vertical',
                 label='ms$^{-1}$',location='left',
                 fraction=0.02, pad=0.1)
   
    # Set the view and 
    plt.legend(bbox_to_anchor=(1.00, 0.25))

    ax1.view_init(elev=30, azim=-140)
    plt.tight_layout()
    plt.show()

    # ============================================================================
    # Plot the raw trajectories
    # ============================================================================
    
    if plot_raw:
        # Initialize the figures
        dlabels = ["Estimated","Desired"]
        fig2, axs2 = plt.subplots(3, 2, figsize=(10, 8))
        fig3, axs3 = plt.subplots(4, 2, figsize=(10, 8))
        fig4, axs4 = plt.subplots(5, 1, figsize=(10, 8))

        # Plot Positions
        ylabels = ["$p_x$","$p_y$","$p_z$"]
        for i in range(3):
            for idx,flight in enumerate(flights):
                axs2[i,0].plot(flight['Tact'],flight['Xact'][i,:],color='tab:blue',alpha=0.5,label=dlabels[0] if idx == 0 else None)
            axs2[i,0].plot(flight['Tref'],flight['Xref'][i,:], color='k', linestyle='--',linewidth=0.8,label=dlabels[1])
            axs2[i,0].set_ylabel(ylabels[i])
            
            if i == 0:
                axs2[i,0].set_ylim([-8,8])
            elif i==1:
                axs2[i,0].set_ylim([-3,3])
            elif i==2:
                axs2[i,0].set_ylim([0,-3])

        axs2[0, 0].set_title('Position')
        axs2[2, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, ncol=5)
        
        # Plot Velocities
        ylabels = ["$v_x$","$v_y$","$v_z$"]

        for i in range(3):
            for idx,flight in enumerate(flights):
                axs2[i,1].plot(flight['Tact'],flight['Xact'][i+3,:],color='tab:blue',alpha=0.5,label=dlabels[0] if idx == 0 else None)
            axs2[i,1].plot(flight['Tref'],flight['Xref'][i+3,:],color='k', linestyle='--',linewidth=0.8,label=dlabels[1])
            axs2[i,1].set_ylabel(ylabels[i])
        
            axs2[i,1].set_ylim([-3,3])
        
        axs2[0, 1].set_title('Velocity')
        axs2[2, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, ncol=5)

        # Plot Orientation
        ylabels = ["$q_x$","$q_y$","$q_z$","q_w"]

        for i in range(4):
            for idx,flight in enumerate(flights):
                axs3[i,0].plot(flight['Tact'],flight['Xact'][i+6,:],color='tab:blue',alpha=0.5,label=dlabels[0] if idx == 0 else None)
            axs3[i,0].plot(flight['Tref'],flight['Xref'][i+6,:],color='k', linestyle='--',linewidth=0.8,label=dlabels[1])
            axs3[i,0].set_ylabel(ylabels[i])
            
            axs3[i,0].set_ylim([-1.2,1.2])
        
        axs3[0, 0].set_title('Orientation')
        axs3[3, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, ncol=5)

                # Plot Inputs
        if baseline:
            ylabels = ['vx', 'vy', 'none','yaw_rate']
            for i in range(4):
                for idx,flight in enumerate(flights):
                    axs3[i,1].plot(flight['Tact'],flight['Xact'][i+3,:],color='tab:blue',alpha=0.5,label=dlabels[0] if idx == 0 else None)
                    # print(flight['Uact'][i,:])
                axs3[i,1].plot(flight['Tact'],flight['Uact'][i,:],color='k', linestyle='--',linewidth=0.8,label=dlabels[1])
                axs3[i,1].set_ylabel(ylabels[i])
                
                if i == 2:
                    axs3[i,1].set_ylim([ -1.0,1.0])
                else:
                    axs3[i,1].set_ylim([-1.0, 1.0])

            axs3[0,1].set_title('Velocities')
            axs3[3,1].legend(["Actual","Inputs"],loc='upper center', bbox_to_anchor=(0.5, -0.2),
                fancybox=True, ncol=5)
        else:
            ylabels = ["$f_n$","$\omega_x$","$\omega_y$","$\omega_z$"]

            for i in range(4):
                for idx,flight in enumerate(flights):
                    axs3[i,1].plot(flight['Tact'],flight['Uact'][i,:],color='tab:blue',alpha=0.5,label=dlabels[0] if idx == 0 else None)
                    # print(flight['Uact'][i,:])
                axs3[i,1].plot(flight['Tref'],flight['Xref'][i,:],color='k', linestyle='--',linewidth=0.8,label=dlabels[1])
                axs3[i,1].set_ylabel(ylabels[i])
                
                if i == 0:
                    axs3[i,1].set_ylim([ 0.0,-1.2])
                else:
                    axs3[i,1].set_ylim([-3.0, 3.0])

            axs3[0,1].set_title('Inputs')
            axs3[3,1].legend(["Actual","Desired"],loc='upper center', bbox_to_anchor=(0.5, -0.2),
                fancybox=True, ncol=5)
            
        # Plot Control Time
        ylabels = ["$Observe$","Orient","Decide","Act","Full Policy"]
        for i in range(5):
            for idx,flight in enumerate(flights):
                axs4[i].plot(flight['Tact'],flight['Tsol'][i,:],color='tab:blue',alpha=0.5,label=dlabels[0] if idx == 0 else None)
            axs4[i].set_ylabel(ylabels[i])

def review_flight(folder:str,
                  Nfiles:Union[None,int]=None,
                  dt_trim:float=0.0,land_check:bool=True,
                  PR_thresh:float=0.3,
                  plot_raw:bool=False,
                  Nrnd:int=3):
    """
    Review the flight data from a folder. We do the following to the data:
    1. Preprocess the data:
        a) detect landings and pad the data accordingly
        b) trim the data based on the given time
    2. Compute the tracking error metrics:
        a) Trajectory Tracking Error (TTE)
        b) Proximity Percentile (PP)
        c) Total Distance Traveled (TDT)
    3. Compute the effective control frequency.
    3. Plot the 3D Trajectories 
    4. Plot the raw data (optional)

    """

    # ============================================================================
    # Preprocess the data
    # ============================================================================

    flights = preprocess_trajectory_data(folder,Nfiles,dt_trim,land_check=land_check)

    # ============================================================================
    # Compute the tracking error metrics
    # ============================================================================

    TTEs,PRs,TDTs = [],[],[]
    for flight in flights:
        TTE = compute_TTE(flight['Xact'],flight['Xref'])
        TTEs.append(TTE)

        PR = compute_PP(flight['Xact'],flight['Xref'],thresh=PR_thresh)
        PRs.append(PR)

        TDT = compute_TDT(flight['Xact'])
        TDTs.append(TDT)

    TDT_ref = len(flights)*compute_TDT(flights[0]['Xref'])

    print("Indv. TTE (m): ",[np.around(np.mean(TTE),Nrnd) for TTE in TTEs])
    print("Total TTE (m): ",np.around(np.mean(np.hstack(TTEs)),Nrnd))
    print("Indv. PR  (%): ",[np.around(PR,Nrnd) for PR in PRs])
    print("Total PR  (%): ",np.around(np.mean(PRs),Nrnd))
    print("Indv. TDT (m): ",[np.around(TDT,Nrnd) for TDT in TDTs])
    print("Total TDT (m): ",np.around(np.sum(TDTs),Nrnd),'of',np.around(TDT_ref,Nrnd))
    
    # ============================================================================
    # Compute the effective control frequency
    # ============================================================================

    Tsols = []
    for flight in flights:
        Tsols.append(flight['Tsol'][4,:])
    print("Effective Control Frequency (Hz): ",[np.around(1/np.mean(Tsol),Nrnd) for Tsol in Tsols])

    # ============================================================================
    # Plot the 3D Trajectories
    # ============================================================================

    # Plot limits
    plim = np.array([
        [ -5.0,  5.0],
        [ -3.5,  3.5],
        [  0.0, -3.0]])
    
    xlim = plim[0,:]
    ylim = plim[1,:]
    zlim = plim[2,:]
    ratio = plim[:,1]-plim[:,0]
        
    # Get the relevant data from flights (trajectories and speed spectrum variables)
    XXact,Spd = [],[]
    for flight in flights:
        XXact.append(flight['Xact'])
        Spd.append(np.linalg.norm(flight['Xact'][3:6,:],axis=0))

    # Determine the colormap
    sdp_all = np.hstack(Spd)
    norm = plt.Normalize(sdp_all.min(), sdp_all.max())
    cmap = plt.get_cmap('viridis')

    # Get the reference trajectory
    Xref = flights[0]['Xref']

    # Initialize the figure
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_zlim(zlim)
    ax1.invert_yaxis()
    ax1.invert_zaxis()
    ax1.set_box_aspect(ratio)  # aspect ratio is 1:1:1 in data space

    # Plot the reference trajectory
    ax1.plot(Xref[0,:],Xref[1,:],Xref[2,:],label='Desired',color='#4d4d4d',linestyle='--')

    # Extract and plot the speed lines
    for Xact,spd in zip(XXact,Spd):
        xyz = Xact[0:3,:].T
        segments = np.stack([xyz[:-1,:], xyz[1:,:]], axis=1)
        lc = Line3DCollection(segments, alpha=0.5,linewidths=2, colors=cmap(norm(spd)))    #     segments = np.concatenate([points[:-1], points[1:]], axis=1).T
        ax1.add_collection(lc)
    ax1.plot([], [], [], label='Actual',color='#3b528b', alpha=0.6)

    # Add the colorbar
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax1,orientation='vertical',
                 label='ms$^{-1}$',location='left',
                 fraction=0.02, pad=0.1)
   
    # Set the view and 
    plt.legend(bbox_to_anchor=(1.00, 0.25))

    ax1.view_init(elev=30, azim=-140)
    plt.tight_layout()
    plt.show()

    # ============================================================================
    # Plot the raw trajectories
    # ============================================================================
    
    if plot_raw:
        # Initialize the figures
        dlabels = ["Estimated","Desired"]
        fig2, axs2 = plt.subplots(3, 2, figsize=(10, 5))
        fig3, axs3 = plt.subplots(4, 2, figsize=(10, 5))
        fig4, axs4 = plt.subplots(5, 1, figsize=(10, 5))

        # Plot Positions
        ylabels = ["$p_x$","$p_y$","$p_z$"]
        for i in range(3):
            for idx,flight in enumerate(flights):
                axs2[i,0].plot(flight['Tact'],flight['Xact'][i,:],color='tab:blue',alpha=0.5,label=dlabels[0] if idx == 0 else None)
            axs2[i,0].plot(flight['Tref'],flight['Xref'][i,:], color='k', linestyle='--',linewidth=0.8,label=dlabels[1])
            axs2[i,0].set_ylabel(ylabels[i])
            
            if i == 0:
                axs2[i,0].set_ylim([-8,8])
            elif i==1:
                axs2[i,0].set_ylim([-3,3])
            elif i==2:
                axs2[i,0].set_ylim([0,-3])

        axs2[0, 0].set_title('Position')
        axs2[2, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, ncol=5)
        
        # Plot Velocities
        ylabels = ["$v_x$","$v_y$","$v_z$"]

        for i in range(3):
            for idx,flight in enumerate(flights):
                axs2[i,1].plot(flight['Tact'],flight['Xact'][i+3,:],color='tab:blue',alpha=0.5,label=dlabels[0] if idx == 0 else None)
            axs2[i,1].plot(flight['Tref'],flight['Xref'][i+3,:],color='k', linestyle='--',linewidth=0.8,label=dlabels[1])
            axs2[i,1].set_ylabel(ylabels[i])
        
            axs2[i,1].set_ylim([-3,3])
        
        axs2[0, 1].set_title('Velocity')
        axs2[2, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, ncol=5)

        # Plot Orientation
        ylabels = ["$q_x$","$q_y$","$q_z$","q_w"]

        for i in range(4):
            for idx,flight in enumerate(flights):
                axs3[i,0].plot(flight['Tact'],flight['Xact'][i+6,:],color='tab:blue',alpha=0.5,label=dlabels[0] if idx == 0 else None)
            axs3[i,0].plot(flight['Tref'],flight['Xref'][i+6,:],color='k', linestyle='--',linewidth=0.8,label=dlabels[1])
            axs3[i,0].set_ylabel(ylabels[i])
            
            axs3[i,0].set_ylim([-1.2,1.2])
        
        axs3[0, 0].set_title('Orientation')
        axs3[3, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, ncol=5)

        # Plot Inputs
        ylabels = ["$f_n$","$\omega_x$","$\omega_y$","$\omega_z$"]

        for i in range(4):
            for idx,flight in enumerate(flights):
                axs3[i,1].plot(flight['Tact'],flight['Uact'][i,:],color='tab:blue',alpha=0.5,label=dlabels[0] if idx == 0 else None)
            axs3[i,1].plot(flight['Tref'],flight['Xref'][i,:],color='k', linestyle='--',linewidth=0.8,label=dlabels[1])
            axs3[i,1].set_ylabel(ylabels[i])
            
            if i == 0:
                axs3[i,1].set_ylim([ 0.0,-1.2])
            else:
                axs3[i,1].set_ylim([-3.0, 3.0])

        axs3[0,1].set_title('Inputs')
        axs3[3,1].legend(["Actual","Desired"],loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, ncol=5)
            
        # Plot Control Time
        ylabels = ["$Observe$","Orient","Decide","Act","Full Policy"]
        for i in range(5):
            for idx,flight in enumerate(flights):
                axs4[i].plot(flight['Tact'],flight['Tsol'][i,:],color='tab:blue',alpha=0.5,label=dlabels[0] if idx == 0 else None)
            axs4[i].set_ylabel(ylabels[i])

def quad_frame(x:np.ndarray,ax:plt.Axes,scale:float=1.0):
    """
    Plot a quadcopter frame in 3D.
    """
    frame_body = scale*np.diag([0.6,0.6,-0.2])
    frame_labels = ["red","green","blue"]
    pos  = x[0:3]
    quat = x[6:10]
    
    for j in range(0,3):
        Rj = R.from_quat(quat).as_matrix()
        arm = Rj@frame_body[j,:]

        frame = np.zeros((3,2))
        if (j == 2):
            frame[:,0] = pos
        else:
            frame[:,0] = pos - arm

        frame[:,1] = pos + arm

        ax.plot(frame[0,:],frame[1,:],frame[2,:], frame_labels[j],label='_nolegend_')

def extract_video(flight_data_path:str,video_path:str):
    """
    Extract the video from the flight data.
    """
    # Load the data
    data = torch.load(flight_data_path,weights_only=False)

    # Unpack the images and process
    Imgs = data['Imgs'].permute(0, 2, 3, 1).numpy()
    Imgs = (Imgs * 255).clip(0, 255).astype(np.uint8)
    
    # Write the video
    write_video(video_path+'.mp4',Imgs,5)