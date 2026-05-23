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

def preprocess_trajectory_data(folder:str,query_name:str,method_type:str,Nfiles:Union[None,int]=None,
                                dt_trim:float=0.0,land_check:bool=True,z_floor:float=0.0):
    # ============================================================================
    # Unpack the data
    # ============================================================================
    
    # Intelligently construct path using query_name and method_type
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path = os.path.join(workspace_path,"cohorts",folder)
    
    # Parse query_name and method_type to construct specific subdirectory
    # Example: query_name="drill_2", method_type="baseline" -> "processed_experiments_drill/baseline_drill_2"
    if query_name and method_type:
        experiment_type = query_name.split("_")[0]  # extract "drill" from "drill_2" 
        folder_path = os.path.join(cohort_path, 'flight_data', f'processed_experiments_{experiment_type}', f'{method_type}_{query_name}')
    else:
        # Fallback to original behavior
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

def _detect_major_reversal(X, threshold_angle_deg=90):
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
                query_name:str,
                method:str,
                query_location,
                initial_location_and_orientation:Union[None,np.ndarray]=None,
                Nfiles:Union[None,int]=None,
                dt_trim:float=0.0,Nrnd:int=3,
                land_check:bool=False,
                plot_raw:bool=True,
                strip_idle:bool=False,
                filter_manual:bool=False,
                filter_velocity:bool=False,
                filter_altitude:bool=False,
                filter_position:bool=False,
                reset_z:bool=False,
                baseline:bool=False,
                ):
    flights = preprocess_trajectory_data(cohort,query_name,method,Nfiles,dt_trim,land_check=land_check)

    # ============================================================================
    # Unpack the data
    # ============================================================================
    # Intelligently construct path using query_name and method_type from flights processing
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path = os.path.join(workspace_path,"cohorts",cohort)
    
    # Parse query_name and method_type to construct specific subdirectory
    # Example: query_name="drill_2", method="baseline" -> "processed_experiments_drill/baseline_drill_2"
    if query_name and method:
        experiment_type = query_name.split("_")[0]  # extract "drill" from "drill_2" 
        folder_path = os.path.join(cohort_path, 'flight_data', f'processed_experiments_{experiment_type}', f'{method}_{query_name}')
    else:
        # Fallback to original behavior
        folder_path = os.path.join(cohort_path,'flight_data')
    
    data_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]
    data_files.sort()

    # Trim file count if need be
    if Nfiles is not None:
        data_files = data_files[-Nfiles:]
    
    print("==============================||   Flight Data Loader   ||==============================||")
    print("File Name :",cohort)
    print("File Count:",len(data_files))
    print("Trim Time :",dt_trim)
    print("Land Check:",land_check)

    fnames = []
    for file in data_files:
        fname = os.path.basename(file)
        fnames.append(fname)
        print("Loading file:",fname)

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

    if filter_manual:
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
    
    if filter_velocity:
        # Detect large velocity direction changes by working backwards from the end
        for f in flights:
            Xact = f.get('Xact', None)
            if Xact is None or Xact.shape[1] < 3:
                continue
            
            # Extract velocity data (rows 3:6 are velocities)
            vel = Xact[3:6, :]  # velocity in x, y, z
            vel_norm = np.linalg.norm(vel, axis=0)  # velocity magnitude at each timestep
            vel_unit = np.zeros_like(vel)
            mask = vel_norm > 1e-6
            vel_unit[:, mask] = vel[:, mask] / vel_norm[mask]

            # Work backwards from the end to find large velocity direction changes
            threshold_angle_deg = 75  # degrees - adjust as needed (90° = sharp turn, 120° = reversal)
            crop_idx = None
            
            # Start from the end and work backwards
            for i in range(vel.shape[1] - 1, 0, -1):  # from end to beginning
                if vel_norm[i] > 1e-6 and vel_norm[i-1] > 1e-6:  # both velocities must be non-zero
                    # Compute angle between consecutive velocity vectors
                    dot = np.sum(vel_unit[:, i] * vel_unit[:, i-1])
                    dot = np.clip(dot, -1.0, 1.0)
                    angle = np.arccos(dot) * 180.0 / np.pi
                    
                    if angle > threshold_angle_deg:
                        crop_idx = i  # crop at the point where the direction change begins
                        print(f"Large velocity direction change detected at index {i}, angle={angle:.1f}°, cropping trajectory")
                        break
            
            # Crop trajectory if needed
            if crop_idx is not None:
                # Crop all time-dependent arrays
                for k in ['Tact', 'Tref']:
                    if k in f and f[k] is not None and f[k].ndim == 1 and f[k].shape[0] > crop_idx:
                        f[k] = f[k][:crop_idx]
                for k in ['Xact', 'Xref', 'Xest', 'Xext', 'Uact', 'Uref']:
                    if k in f and f[k] is not None and f[k].ndim == 2 and f[k].shape[1] > crop_idx:
                        f[k] = f[k][:, :crop_idx]
                if 'Tsol' in f and f['Tsol'] is not None and f['Tsol'].ndim == 2 and f['Tsol'].shape[1] > crop_idx:
                    f['Tsol'] = f['Tsol'][:, :crop_idx]
        
    if filter_altitude:
        # Detect large altitude drops by working backwards from the end
        for f, fname in zip(flights, fnames):
            if "onboard" in fname:
                Xref = f.get('Xref', None)
                if Xref is None or Xref.shape[1] < 3:
                    continue
                
                # Extract altitude data (row 2 is z position)
                z = Xref[2, :]  # altitude
                query_z = query_location[2]
                threshold_drop = 0.99 * abs(query_z)  # 90% of absolute query altitude
                crop_idx = None
                
                # Work backwards from the end to find large altitude changes
                final_z = z[-1]  # altitude at the end of trajectory
                
                for i in range(z.shape[0] - 2, -1, -1):  # from second-to-last to beginning
                    cumulative_change = abs(z[i] - final_z)  # cumulative change from end
                    
                    if cumulative_change > threshold_drop:
                        crop_idx = i + 1  # crop just after this point to preserve good data
                        direction = "drop" if z[i] > final_z else "increase"
                        print(f"Large cumulative altitude {direction} detected from index {i} to end, total change={cumulative_change:.2f}m (>{threshold_drop:.2f}m), cropping trajectory")
                        break
                
                # Crop trajectory if needed
                if crop_idx is not None:
                    # Crop all time-dependent arrays
                    for k in ['Tact', 'Tref']:
                        if k in f and f[k] is not None and f[k].ndim == 1 and f[k].shape[0] > crop_idx:
                            f[k] = f[k][:crop_idx]
                    for k in ['Xact', 'Xref', 'Xest', 'Xext', 'Uact', 'Uref']:
                        if k in f and f[k] is not None and f[k].ndim == 2 and f[k].shape[1] > crop_idx:
                            f[k] = f[k][:, :crop_idx]
                    if 'Tsol' in f and f['Tsol'] is not None and f['Tsol'].ndim == 2 and f['Tsol'].shape[1] > crop_idx:
                        f['Tsol'] = f['Tsol'][:, :crop_idx]
            else:
                Xact = f.get('Xact', None)
                if Xact is None or Xact.shape[1] < 3:
                    continue
                
                # Extract altitude data (row 2 is z position)
                z = Xact[2, :]  # altitude
                query_z = query_location[2]
                threshold_drop = 0.9 * abs(query_z)  # 90% of absolute query altitude
                crop_idx = None
                
                # Work backwards from the end to find large altitude changes
                final_z = z[-1]  # altitude at the end of trajectory
                
                for i in range(z.shape[0] - 2, -1, -1):  # from second-to-last to beginning
                    cumulative_change = abs(z[i] - final_z)  # cumulative change from end
                    
                    if cumulative_change > threshold_drop:
                        crop_idx = i + 1  # crop just after this point to preserve good data
                        direction = "drop" if z[i] > final_z else "increase"
                        print(f"Large cumulative altitude {direction} detected from index {i} to end, total change={cumulative_change:.2f}m (>{threshold_drop:.2f}m), cropping trajectory")
                        break
                
                # Crop trajectory if needed
                if crop_idx is not None:
                    # Crop all time-dependent arrays
                    for k in ['Tact', 'Tref']:
                        if k in f and f[k] is not None and f[k].ndim == 1 and f[k].shape[0] > crop_idx:
                            f[k] = f[k][:crop_idx]
                    for k in ['Xact', 'Xref', 'Xest', 'Xext', 'Uact', 'Uref']:
                        if k in f and f[k] is not None and f[k].ndim == 2 and f[k].shape[1] > crop_idx:
                            f[k] = f[k][:, :crop_idx]
                    if 'Tsol' in f and f['Tsol'] is not None and f['Tsol'].ndim == 2 and f['Tsol'].shape[1] > crop_idx:
                        f['Tsol'] = f['Tsol'][:, :crop_idx]

    if filter_position:
        # Detect XY position direction reversals using angle-based detection
        for f, fname in zip(flights, fnames):
            if "onboard" in fname:
                X = f.get('Xref', None)
                if X is None or X.shape[1] < 3:
                    continue
            else:
                X = f.get('Xact', None)
                if X is None or X.shape[1] < 3:
                    continue
            
            # Use XY position only (rows 0:2) for direction reversal detection
            xy_pos = X[0:2, :]  # Extract x,y positions only
            
            # Use a windowed approach to detect sustained direction changes
            window_size = 20  # samples - adjust based on data frequency and desired sensitivity
            threshold_angle_deg = 120  # degrees - adjust as needed for XY direction reversals
            crop_idx = None
            
            # Need at least 2*window_size samples to compute windowed angles
            if xy_pos.shape[1] >= 2 * window_size:
                # Work backwards from the end to find direction reversals
                for i in range(xy_pos.shape[1] - 2 * window_size - 1, -1, -1):
                    # Compute velocity vector over the first window
                    start_pos = xy_pos[:, i]
                    mid_pos = xy_pos[:, i + window_size]
                    end_pos = xy_pos[:, i + 2 * window_size]
                    
                    # Two windowed velocity vectors
                    v1 = mid_pos - start_pos  # first window velocity
                    v2 = end_pos - mid_pos    # second window velocity
                    
                    # Normalize velocities
                    v1_norm = np.linalg.norm(v1)
                    v2_norm = np.linalg.norm(v2)
                    
                    if v1_norm > 1e-6 and v2_norm > 1e-6:  # avoid division by zero
                        v1_unit = v1 / v1_norm
                        v2_unit = v2 / v2_norm
                        
                        # Compute angle between windowed velocity vectors
                        dot = np.dot(v1_unit, v2_unit)
                        dot = np.clip(dot, -1.0, 1.0)
                        angle = np.arccos(dot) * 180.0 / np.pi
                        
                        if angle > threshold_angle_deg:
                            crop_idx = i + window_size  # crop at the midpoint where direction change occurs
                            reversal_pos = xy_pos[:, crop_idx] if crop_idx < xy_pos.shape[1] else xy_pos[:, -1]
                            print(f"XY direction reversal detected at index {crop_idx}, position {reversal_pos}, windowed angle={angle:.1f}° (>{threshold_angle_deg}°), window_size={window_size}, cropping trajectory")
                            break
                
                # Crop trajectory if needed
                if crop_idx is not None:
                    # Crop all time-dependent arrays
                    for k in ['Tact', 'Tref']:
                        if k in f and f[k] is not None and f[k].ndim == 1 and f[k].shape[0] > crop_idx:
                            f[k] = f[k][:crop_idx]
                    for k in ['Xact', 'Xref', 'Xest', 'Xext', 'Uact', 'Uref']:
                        if k in f and f[k] is not None and f[k].ndim == 2 and f[k].shape[1] > crop_idx:
                            f[k] = f[k][:, :crop_idx]
                    if 'Tsol' in f and f['Tsol'] is not None and f['Tsol'].ndim == 2 and f['Tsol'].shape[1] > crop_idx:
                        f['Tsol'] = f['Tsol'][:, :crop_idx]
    
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

    if method == "ssv_onboard":
        S = np.diag([1, -1, -1])

        for f in flights:
            # Transform reference state if present
            if 'Xref' in f and f['Xref'] is not None:
                Xr = f['Xref']
                Xr[0:3, :] = S @ Xr[0:3, :]
                if Xr.shape[0] >= 10:
                    Xr[6:10, :] = np.apply_along_axis(_quat_flu_to_frd_xyzw, 0, Xr[6:10, :])
                f['Xref'] = Xr  # (in-place; assignment optional)

        # Compute and print average starting location and orientation
        start_positions = []
        start_orientations = []
        inferred_yaws = []           # NEW: store per-flight inferred yaw from Xref (rad)
        rotated_start_yaws = []      # NEW: store per-flight rotated Xact start yaw (rad)

        for f in flights:
            Xr = f['Xref']    # already S-transformed & saved above
            Xa = f['Xact']

            start_positions.append(Xr[0:3, 0])

            # --- infer world-frame yaw from earliest XY motion in Xref ---
            yaw = 0.0
            p0 = Xr[0:3, 0]
            max_scan = min(80, Xr.shape[1])
            # prefer a long baseline: start -> last within the cap
            pj = Xr[0:3, max_scan-1]
            disp = pj - p0

            # if that baseline is too short (e.g., hover), fall back to the farthest point
            if np.linalg.norm(disp[:2]) <= 1e-6:
                # argmax over 2D distance from p0 in first max_scan columns
                dxy = Xr[0:2, :max_scan] - p0[0:2, None]
                j = np.argmax(np.linalg.norm(dxy, axis=0))
                disp = Xr[0:3, j] - p0

            yaw = np.arctan2(disp[1], disp[0])
            inferred_yaws.append(yaw)

            print("Ref disp dx,dy:", float(disp[0]), float(disp[1]), 
                " -> yaw_deg:", float(np.degrees(yaw)))

            half = 0.5 * yaw
            ax, ay, az, aw = 0.0, 0.0, np.sin(half), np.cos(half)  # q_align (XYZW)

            # --- rotate all Xact quats into the world frame: q_world = q_align ⊗ q_xact ---
            if Xa.shape[0] >= 10:
                Q = Xa[6:10, :]
                qx, qy, qz, qw = Q[0, :], Q[1, :], Q[2, :], Q[3, :]

                rx = aw*qx + ax*qw + ay*qz - az*qy
                ry = aw*qy - ax*qz + ay*qw + az*qx
                rz = aw*qz + ax*qy - ay*qx + az*qw
                rw = aw*qw - ax*qx - ay*qy - az*qz

                nrm = np.sqrt(rx*rx + ry*ry + rz*rz + rw*rw)
                nrm[nrm == 0] = 1.0
                Q_rot = np.vstack((rx/nrm, ry/nrm, rz/nrm, rw/nrm))

                Xa[6:10, :] = Q_rot
                f['Xact'] = Xa

                start_orientations.append(Q_rot[:, 0])

                # NEW: yaw of the rotated Xact start quat (XYZW -> yaw)
                x, y, z, w = Q_rot[:, 0]
                yaw_start = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
                rotated_start_yaws.append(yaw_start)
            else:
                start_orientations.append(np.full(4, np.nan))
                rotated_start_yaws.append(np.nan)

        print(f"start_positions: {start_positions}")

        avg_start_pos = np.mean(np.stack(start_positions), axis=0)
        avg_start_ori = np.mean(np.stack(start_orientations), axis=0)

        # --- NEW: print yaws (sanity checks) ---
        # Inferred yaw from Xref (global frame), per-flight mean:
        inferred_yaws = np.array(inferred_yaws, dtype=float)
        print("Inferred yaw from Xref (deg) per-flight mean:",
            np.degrees(np.nanmean(inferred_yaws)))

        # Rotated Xact starting yaw (now in global frame), per-flight mean:
        rotated_start_yaws = np.array(rotated_start_yaws, dtype=float)
        print("Rotated Xact start yaw (deg) per-flight mean:",
            np.degrees(np.nanmean(rotated_start_yaws)))

        # Yaw from the averaged quaternion itself (normalize first):
        q = avg_start_ori / (np.linalg.norm(avg_start_ori) if np.linalg.norm(avg_start_ori) > 0 else 1.0)
        x, y, z, w = q
        avg_yaw_from_avg_quat = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        print("Average starting orientation (xyzw, world):", avg_start_ori)
        print("Average starting yaw from avg quat (deg):", np.degrees(avg_yaw_from_avg_quat))

        print("Average starting location:", avg_start_pos)

    elif method == "ssv_mocap":
        # First pass: collect all original starting positions to compute center
        original_starts = []
        for f, fname in zip(flights, fnames):
            if "realign" in fname:
                Xa = f['Xact']
                if Xa.shape[0] >= 10:
                    original_starts.append(Xa[0:3, 0].copy())
        
        # Compute the original center position
        if original_starts:
            original_center = np.mean(np.stack(original_starts), axis=0)
        else:
            original_center = None
        
        # Second pass: apply transformations
        for f,fname in zip(flights, fnames):
            if "cw" in fname:
                Xa = f['Xact']
                Xa[[0, 1], :] = np.vstack(( Xa[1, :], -Xa[0, :] ))   # positions
                Xa[[3, 4], :] = np.vstack(( Xa[4, :], -Xa[3, :] ))   # velocities (optional)
                if Xa.shape[0] >= 10:
                    Q = Xa[6:10, :]
                    # 90 deg CCW about Z axis: quaternion [0, 0, -sin(pi/4), cos(pi/4)] (XYZW)
                    angle = np.pi / 2
                    q_rot = np.array([0.0, 0.0, -np.sin(angle / 2), np.cos(angle / 2)], dtype=float)
                    # Quaternion multiplication: q_rot ⊗ Q
                    qx, qy, qz, qw = Q[0, :], Q[1, :], Q[2, :], Q[3, :]
                    rx = q_rot[3]*qx + q_rot[0]*qw + q_rot[1]*qz - q_rot[2]*qy
                    ry = q_rot[3]*qy - q_rot[0]*qz + q_rot[1]*qw + q_rot[2]*qx
                    rz = q_rot[3]*qz + q_rot[0]*qy - q_rot[1]*qx + q_rot[2]*qw
                    rw = q_rot[3]*qw - q_rot[0]*qx - q_rot[1]*qy - q_rot[2]*qz
                    nrm = np.sqrt(rx*rx + ry*ry + rz*rz + rw*rw)
                    nrm[nrm == 0] = 1.0
                    Xa[6:10, :] = np.vstack((rx/nrm, ry/nrm, rz/nrm, rw/nrm))
                f['Xact'] = Xa
            elif "ccw" in fname:
                Xa = f['Xact']
                Xa[[0, 1], :] = np.vstack(( -Xa[1, :], Xa[0, :] ))
                Xa[[3, 4], :] = np.vstack(( -Xa[4, :], Xa[3, :] ))   # velocities (optional)
                if Xa.shape[0] >= 10:
                    Q = Xa[6:10, :]
                    # 90 deg CCW about Z axis: quaternion [0, 0, -sin(pi/4), cos(pi/4)] (XYZW)
                    angle = np.pi / 2
                    q_rot = np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)], dtype=float)
                    # Quaternion multiplication: q_rot ⊗ Q
                    qx, qy, qz, qw = Q[0, :], Q[1, :], Q[2, :], Q[3, :]
                    rx = q_rot[3]*qx + q_rot[0]*qw + q_rot[1]*qz - q_rot[2]*qy
                    ry = q_rot[3]*qy - q_rot[0]*qz + q_rot[1]*qw + q_rot[2]*qx
                    rz = q_rot[3]*qz + q_rot[0]*qy - q_rot[1]*qx + q_rot[2]*qw
                    rw = q_rot[3]*qw - q_rot[0]*qx - q_rot[1]*qy - q_rot[2]*qz
                    nrm = np.sqrt(rx*rx + ry*ry + rz*rz + rw*rw)
                    nrm[nrm == 0] = 1.0
                    Xa[6:10, :] = np.vstack((rx/nrm, ry/nrm, rz/nrm, rw/nrm))
                f['Xact'] = Xa
            elif "realign" in fname:
                # Inputs you provide (world-frame target pose):
                target_pos  = np.array(initial_location_and_orientation[:3], dtype=float)                  # (3,)
                target_quat = np.array(initial_location_and_orientation[3:], dtype=float)                  # (4,) XYZW
                target_quat = target_quat / (np.linalg.norm(target_quat) or 1.0)

                Xa = f['Xact']
                if Xa.shape[0] < 10:
                    continue
                
                # Initialize rotation matrix as identity (no rotation by default)
                R = np.eye(3, dtype=float)
                reoriented = False
                repositioned = False
                
                # Store original starting position for repositioning offset calculation
                original_start_pos = Xa[0:3, 0].copy()
                
                # Check if we need to reorient (change orientation)
                if "reorient" in fname:
                    # -- 1) per-flight start quat (local/odometry frame), normalize
                    q0 = Xa[6:10, 0].astype(float)
                    q0 = q0 / (np.linalg.norm(q0) or 1.0)                          # XYZW
                    x0, y0, z0, w0 = q0
                    q0_inv = np.array([-x0, -y0, -z0,  w0], dtype=float)           # inverse (unit)

                    # -- 2) delta quat: q_delta = q_target ⊗ q0^{-1}
                    xt, yt, zt, wt = target_quat
                    xi, yi, zi, wi = q0_inv
                    dx = wt*xi + xt*wi + yt*zi - zt*yi
                    dy = wt*yi - xt*zi + yt*wi + zt*xi
                    dz = wt*zi + xt*yi - yt*xi + zt*wi
                    dw = wt*wi - xt*xi - yt*yi - zt*zi
                    q_delta = np.array([dx, dy, dz, dw], dtype=float)
                    q_delta = q_delta / (np.linalg.norm(q_delta) or 1.0)

                    # -- 3) rotate all Xact quats: q' = q_delta ⊗ q_xact  (vectorized)
                    Q  = Xa[6:10, :]
                    qx, qy, qz, qw = Q[0,:], Q[1,:], Q[2,:], Q[3,:]
                    dx, dy, dz, dw = q_delta
                    rx = dw*qx + dx*qw + dy*qz - dz*qy
                    ry = dw*qy - dx*qz + dy*qw + dz*qx
                    rz = dw*qz + dx*qy - dy*qx + dz*qw
                    rw = dw*qw - dx*qx - dy*qy - dz*qz
                    nrm = np.sqrt(rx*rx + ry*ry + rz*rz + rw*rw); nrm[nrm == 0] = 1.0
                    Xa[6:10, :] = np.vstack((rx/nrm, ry/nrm, rz/nrm, rw/nrm))

                    # -- 4) compute rotation matrix from q_delta for position transformation
                    dx,dy,dz,dw = q_delta
                    Rxx = 1 - 2*(dy*dy + dz*dz);  Rxy = 2*(dx*dy - dz*dw);  Rxz = 2*(dx*dz + dy*dw)
                    Ryx = 2*(dx*dy + dz*dw);      Ryy = 1 - 2*(dx*dx + dz*dz);  Ryz = 2*(dy*dz - dx*dw)
                    Rzx = 2*(dx*dz - dy*dw);      Rzy = 2*(dy*dz + dx*dw);      Rzz = 1 - 2*(dx*dx + dy*dy)
                    R = np.array([[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]], float)
                    
                    # -- 5) Apply rotation to positions (always when reorienting)
                    P = Xa[0:3,:]
                    p0 = P[:,0].copy()
                    # Rotate positions around the starting position
                    P_rotated = R @ (P - p0[:,None]) + p0[:,None]
                    Xa[0:3,:] = P_rotated
                    
                    reoriented = True
                    print(f"[{fname}] Applied reorientation transformation")

                # Check if we need to reposition (change position)
                if "repos" in fname:
                    P  = Xa[0:3,:]
                    p0 = P[:,0].copy()

                    # Calculate deviation from original center (preserves relative spread)
                    if original_center is not None:
                        deviation_from_center = original_start_pos - original_center
                    else:
                        deviation_from_center = np.zeros(3)  # fallback if no center computed
                    
                    # If we reoriented, apply the same rotation to the deviation
                    if reoriented:
                        deviation_from_center = R @ deviation_from_center

                    alpha = 1.0   # set to 1.0 to preserve spread, 0.0 to co-locate at target

                    # Translate to target position while preserving relative spread
                    P_aligned = P - p0[:,None] + target_pos[:,None] + alpha * deviation_from_center[:,None]
                    Xa[0:3,:] = P_aligned
                    repositioned = True
                    print(f"[{fname}] Applied repositioning transformation")

                f['Xact'] = Xa

                # Print final state only if any transformations were applied
                if reoriented or repositioned:
                    x,y,z,w = Xa[6:10, 0]
                    yaw_tgt_deg = np.degrees(np.arctan2(2*(target_quat[3]*target_quat[2] + target_quat[0]*target_quat[1]),
                                                        1 - 2*(target_quat[1]**2 + target_quat[2]**2)))
                    yaw_new_deg = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
                    
                    transformations = []
                    if reoriented:
                        transformations.append("reoriented")
                    if repositioned:
                        transformations.append("repositioned")
                    
                    print(f"[{fname}] Final result ({', '.join(transformations)}): start pos -> {Xa[0:3,0]}, start yaw -> {yaw_new_deg:.2f}° (target {yaw_tgt_deg:.2f}°)")

            else:
                Xa = f['Xact']
                f['Xact'] = Xa
    else:
        # First pass: collect all original starting positions to compute center
        original_starts = []
        for f, fname in zip(flights, fnames):
            if "realign" in fname:
                Xa = f['Xact']
                if Xa.shape[0] >= 10:
                    original_starts.append(Xa[0:3, 0].copy())
        
        # Compute the original center position
        if original_starts:
            original_center = np.mean(np.stack(original_starts), axis=0)
        else:
            original_center = None
        
        # Second pass: apply transformations
        for f,fname in zip(flights, fnames):
            if "cw" in fname:
                Xa = f['Xact']
                Xa[[0, 1], :] = np.vstack(( Xa[1, :], -Xa[0, :] ))   # positions
                Xa[[3, 4], :] = np.vstack(( Xa[4, :], -Xa[3, :] ))   # velocities (optional)
                f['Xact'] = Xa
            elif "realign" in fname:
                # Inputs you provide (world-frame target pose):
                target_pos  = np.array(initial_location_and_orientation[:3], dtype=float)                  # (3,)
                target_quat = np.array(initial_location_and_orientation[3:], dtype=float)                  # (4,) XYZW
                target_quat = target_quat / (np.linalg.norm(target_quat) or 1.0)

                Xa = f['Xact']
                if Xa.shape[0] < 10:
                    continue
                
                # Initialize rotation matrix as identity (no rotation by default)
                R = np.eye(3, dtype=float)
                reoriented = False
                repositioned = False
                
                # Store original starting position for repositioning offset calculation
                original_start_pos = Xa[0:3, 0].copy()
                
                # Check if we need to reorient (change orientation)
                if "reorient" in fname:
                    # -- 1) per-flight start quat (local/odometry frame), normalize
                    q0 = Xa[6:10, 0].astype(float)
                    q0 = q0 / (np.linalg.norm(q0) or 1.0)                          # XYZW
                    x0, y0, z0, w0 = q0
                    q0_inv = np.array([-x0, -y0, -z0,  w0], dtype=float)           # inverse (unit)

                    # -- 2) delta quat: q_delta = q_target ⊗ q0^{-1}
                    xt, yt, zt, wt = target_quat
                    xi, yi, zi, wi = q0_inv
                    dx = wt*xi + xt*wi + yt*zi - zt*yi
                    dy = wt*yi - xt*zi + yt*wi + zt*xi
                    dz = wt*zi + xt*yi - yt*xi + zt*wi
                    dw = wt*wi - xt*xi - yt*yi - zt*zi
                    q_delta = np.array([dx, dy, dz, dw], dtype=float)
                    q_delta = q_delta / (np.linalg.norm(q_delta) or 1.0)

                    # -- 3) rotate all Xact quats: q' = q_delta ⊗ q_xact  (vectorized)
                    Q  = Xa[6:10, :]
                    qx, qy, qz, qw = Q[0,:], Q[1,:], Q[2,:], Q[3,:]
                    dx, dy, dz, dw = q_delta
                    rx = dw*qx + dx*qw + dy*qz - dz*qy
                    ry = dw*qy - dx*qz + dy*qw + dz*qx
                    rz = dw*qz + dx*qy - dy*qx + dz*qw
                    rw = dw*qw - dx*qx - dy*qy - dz*qz
                    nrm = np.sqrt(rx*rx + ry*ry + rz*rz + rw*rw); nrm[nrm == 0] = 1.0
                    Xa[6:10, :] = np.vstack((rx/nrm, ry/nrm, rz/nrm, rw/nrm))

                    # -- 4) compute rotation matrix from q_delta for position transformation
                    dx,dy,dz,dw = q_delta
                    Rxx = 1 - 2*(dy*dy + dz*dz);  Rxy = 2*(dx*dy - dz*dw);  Rxz = 2*(dx*dz + dy*dw)
                    Ryx = 2*(dx*dy + dz*dw);      Ryy = 1 - 2*(dx*dx + dz*dz);  Ryz = 2*(dy*dz - dx*dw)
                    Rzx = 2*(dx*dz - dy*dw);      Rzy = 2*(dy*dz + dx*dw);      Rzz = 1 - 2*(dx*dx + dy*dy)
                    R = np.array([[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]], float)
                    
                    # -- 5) Apply rotation to positions (always when reorienting)
                    P = Xa[0:3,:]
                    p0 = P[:,0].copy()
                    # Rotate positions around the starting position
                    P_rotated = R @ (P - p0[:,None]) + p0[:,None]
                    Xa[0:3,:] = P_rotated
                    
                    reoriented = True
                    print(f"[{fname}] Applied reorientation transformation")

                # Check if we need to reposition (change position)
                if "repos" in fname:
                    P  = Xa[0:3,:]
                    p0 = P[:,0].copy()

                    # Calculate deviation from original center (preserves relative spread)
                    if original_center is not None:
                        deviation_from_center = original_start_pos - original_center
                    else:
                        deviation_from_center = np.zeros(3)  # fallback if no center computed
                    
                    # If we reoriented, apply the same rotation to the deviation
                    if reoriented:
                        deviation_from_center = R @ deviation_from_center

                    alpha = 1.0   # set to 1.0 to preserve spread, 0.0 to co-locate at target

                    # Translate to target position while preserving relative spread
                    P_aligned = P - p0[:,None] + target_pos[:,None] + alpha * deviation_from_center[:,None]
                    Xa[0:3,:] = P_aligned
                    repositioned = True
                    print(f"[{fname}] Applied repositioning transformation")

                f['Xact'] = Xa

                # Print final state only if any transformations were applied
                if reoriented or repositioned:
                    x,y,z,w = Xa[6:10, 0]
                    yaw_tgt_deg = np.degrees(np.arctan2(2*(target_quat[3]*target_quat[2] + target_quat[0]*target_quat[1]),
                                                        1 - 2*(target_quat[1]**2 + target_quat[2]**2)))
                    yaw_new_deg = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
                    
                    transformations = []
                    if reoriented:
                        transformations.append("reoriented")
                    if repositioned:
                        transformations.append("repositioned")
                    
                    print(f"[{fname}] Final result ({', '.join(transformations)}): start pos -> {Xa[0:3,0]}, start yaw -> {yaw_new_deg:.2f}° (target {yaw_tgt_deg:.2f}°)")

            else:
                Xa = f['Xact']
                f['Xact'] = Xa
    
    # else:
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
        # print(flight['Xact'][3:6,:])
        Spd.append(np.linalg.norm(flight['Xact'][3:6,:],axis=0))

    # Determine the colormap
    sdp_all = np.hstack(Spd)
    norm = plt.Normalize(sdp_all.min(), sdp_all.max())
    cmap = plt.get_cmap('viridis')

    XXref = []
    for f in flights:
        if 'Xref' in f and f['Xref'] is not None:
            XXref.append(f['Xref'])
        else:
            XXref.append(None)
    
    # Initialize the figure
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Add figure title showing query and method
    fig1.suptitle(f'3D Trajectory - Query: {query_name} | Method: {method}', fontsize=12, fontweight='bold')

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_zlim(zlim)
    ax1.invert_yaxis()
    ax1.invert_zaxis()
    ax1.set_box_aspect(ratio)  # aspect ratio is 1:1:1 in data space

    if method != "ssv_onboard":
        # Extract and plot Xact and the speed lines
        for Xact,spd in zip(XXact,Spd):
            xyz = Xact[0:3,:].T
            segments = np.stack([xyz[:-1,:], xyz[1:,:]], axis=1)
            lc = Line3DCollection(segments, alpha=0.5,linewidths=2, colors=cmap(norm(spd)))    #     segments = np.concatenate([points[:-1], points[1:]], axis=1).T
            ax1.add_collection(lc)
        #NOTE COMMENT THIS OUT IF PLOTS ARE NOT WORKING
            # N = xyz.shape[0]
            # if N >= 2 and Xact.shape[0] >= 10:
            #     # choose up to ~20 evenly spaced indices
            #     max_arrows = 10
            #     idxs = np.linspace(0, N-1, num=min(max_arrows, N), dtype=int)

            #     # positions
            #     Px = xyz[idxs, 0]
            #     Py = xyz[idxs, 1]
            #     Pz = xyz[idxs, 2]

            #     # quaternions: assumed order [qx, qy, qz, qw]
            #     quats = Xact[6:10, idxs].T  # shape (K,4)

            #     # normalize quaternions (guard against NaNs/zeros)
            #     norms = np.linalg.norm(quats, axis=1, keepdims=True)
            #     norms[norms == 0] = 1.0
            #     quats = quats / norms

            #     # convert to forward vectors using SciPy
            #     # SciPy expects [x, y, z, w]
            #     from scipy.spatial.transform import Rotation as R
            #     Rm = R.from_quat(quats)
            #     # local +x axis as "forward" direction
            #     dirs = Rm.apply(np.array([1.0, 0.0, 0.0]))  # shape (K,3)

            #     # pick a reasonable arrow length based on step size of the path
            #     # (robust to units and sampling rate)
            #     steps = np.diff(xyz, axis=0)
            #     step_med = np.median(np.linalg.norm(steps, axis=1)) if steps.size else 1.0
            #     arrow_len = max(1e-6, 10.0 * step_med)  # tune 0.5 if you want longer/shorter arrows

            #     # one batched quiver call (MUCH faster & safer than looping)
            #     ax1.quiver(
            #         Px, Py, Pz,
            #         dirs[:, 0], dirs[:, 1], dirs[:, 2],
            #         length=arrow_len,
            #         normalize=True,          # scales all arrows to 'length'
            #         pivot='tail',            # put tail at position
            #         linewidths=0.8,
            #         color='orange',
            #         alpha=0.9,
            #         zorder=3,
            #     )
            #
        ax1.plot([], [], [], label='Xact',color='#3b528b', alpha=0.6)
        quad_frame(XXact[0][:, 0], ax1, scale=1.0)

        # Add the colorbar
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax1,orientation='vertical',
                    label='ms$^{-1}$',location='left',
                    fraction=0.02, pad=0.1)
    
    else:
        # Extract and plot Xref
        for Xref,spd in zip(XXref,Spd):
            xyz = Xref[0:3,:].T
            segments = np.stack([xyz[:-1,:], xyz[1:,:]], axis=1)
            lc = Line3DCollection(segments, alpha=0.5,linewidths=2, color='black',linestyle='--')#colors=cmap(norm(spd)))    #     segments = np.concatenate([points[:-1], points[1:]], axis=1).T
            ax1.add_collection(lc)
        #NOTE COMMENT THIS OUT IF PLOTS ARE NOT WORKING
            # N = xyz.shape[0]
            # if N >= 2 and Xact.shape[0] >= 10:
            #     # choose up to ~20 evenly spaced indices
            #     max_arrows = 10
            #     idxs = np.linspace(0, N-1, num=min(max_arrows, N), dtype=int)

            #     # positions
            #     Px = xyz[idxs, 0]
            #     Py = xyz[idxs, 1]
            #     Pz = xyz[idxs, 2]

            #     # quaternions: assumed order [qx, qy, qz, qw]
            #     quats = Xact[6:10, idxs].T  # shape (K,4)

            #     # normalize quaternions (guard against NaNs/zeros)
            #     norms = np.linalg.norm(quats, axis=1, keepdims=True)
            #     norms[norms == 0] = 1.0
            #     quats = quats / norms

            #     # convert to forward vectors using SciPy
            #     # SciPy expects [x, y, z, w]
            #     from scipy.spatial.transform import Rotation as R
            #     Rm = R.from_quat(quats)
            #     # local +x axis as "forward" direction
            #     dirs = Rm.apply(np.array([1.0, 0.0, 0.0]))  # shape (K,3)

            #     # pick a reasonable arrow length based on step size of the path
            #     # (robust to units and sampling rate)
            #     steps = np.diff(xyz, axis=0)
            #     step_med = np.median(np.linalg.norm(steps, axis=1)) if steps.size else 1.0
            #     arrow_len = max(1e-6, 10.0 * step_med)  # tune 0.5 if you want longer/shorter arrows

            #     # one batched quiver call (MUCH faster & safer than looping)
            #     ax1.quiver(
            #         Px, Py, Pz,
            #         dirs[:, 0], dirs[:, 1], dirs[:, 2],
            #         length=arrow_len,
            #         normalize=True,          # scales all arrows to 'length'
            #         pivot='tail',            # put tail at position
            #         linewidths=0.8,
            #         color='orange',
            #         alpha=0.9,
            #         zorder=3,
            #     )
            #
        ax1.plot([], [], [], label='Xref',color="#000000", alpha=0.6)
        # quad_frame(XXref[0][:, 0], ax1, scale=1.0)
        # ax1.plot(Xref[0,:],Xref[1,:],Xref[2,:],label='Xref',color='#4d4d4d',linestyle='--')
    
    # Plot the query location as a dot
    ax1.scatter(query_location[0], query_location[1], query_location[2], color='red', s=60, label='Query Location', zorder=10)
    # Plot a circle of radius 1.5m around the query location (in the XY plane)
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = (query_location[0] + 1.5 * np.cos(theta))
    circle_y = (query_location[1] + 1.5 * np.sin(theta))
    circle_z = np.full_like(theta, query_location[2])
    ax1.plot(circle_x, circle_y, circle_z, color='red', linestyle=':', linewidth=2, label='1.5m Radius')

    all_query_in_views_fov = []   # list of boolean arrays, one per trajectory
    all_final_yaws_deg    = []   # list of float arrays (deg), one per trajectory
    end_yaw_error_deg     = []   # list of scalars (deg), one per trajectory

    fov_half_deg = 42.5
    goal_radius = 2.00
    goal_radius_1 = 1.5*goal_radius
    if method == "ssv_onboard":
        # Compute and print the normalized distance to the boundary of the exclusion zone for all trajectories
        for idx, (Xref, Xact) in enumerate(zip(XXref, XXact)):
            if Xref.shape[1] == 0:
                continue
            goal_pos = np.array(query_location[:2])
            final_pos_xy = Xref[0:2, -1]
            final_dist_to_center = np.linalg.norm(final_pos_xy - goal_pos)
            
            # Check if final position is inside the radius
            if final_dist_to_center < goal_radius:
                traj_xy = Xref[0:2, :]
                dists_to_center = np.linalg.norm(traj_xy - goal_pos[:, None], axis=0)
                boundary_idx = np.argmin(np.abs(dists_to_center - goal_radius))
                eval_pos_xy = traj_xy[:, boundary_idx]
                dist_to_center = dists_to_center[boundary_idx]
                final_quat = Xact[6:10, boundary_idx]
                idx_end = boundary_idx
            else:
                eval_pos_xy = final_pos_xy
                dist_to_center = final_dist_to_center
                final_quat = Xact[6:10, -1]
                idx_end = Xact.shape[1] - 1

            dist_initial = np.linalg.norm(Xref[0:2, 0] - goal_pos)
            dist_to_boundary = abs(dist_to_center - goal_radius)
            dist_to_outer_boundary = abs(dist_to_center - goal_radius_1)

            # Compute yaw error arrays throughout the trajectory up until idx_end
            final_yaws_deg = []
            query_in_views_fov = []
            traj_xy = Xref[0:2, :]
            for t in range(idx_end + 1):
                quat_t = Xact[6:10, t]
                qx, qy, qz, qw = quat_t
                actual_yaw_t = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
                pos_t = traj_xy[:, t]
                goal_direction_t = goal_pos - pos_t
                if np.linalg.norm(goal_direction_t) > 1e-6:
                    required_yaw_t = np.arctan2(goal_direction_t[1], goal_direction_t[0])
                    yaw_error_t = abs(actual_yaw_t - required_yaw_t)
                    if yaw_error_t > np.pi:
                        yaw_error_t = 2 * np.pi - yaw_error_t
                    yaw_err_deg = np.degrees(yaw_error_t)
                else:
                    yaw_err_deg = 0.0
                final_yaws_deg.append(yaw_err_deg)
                query_in_views_fov.append(abs(yaw_err_deg) <= fov_half_deg)

            final_yaws_deg = np.array(final_yaws_deg, dtype=float)
            query_in_views_fov = np.array(query_in_views_fov, dtype=bool)

            all_final_yaws_deg.append(final_yaws_deg)
            all_query_in_views_fov.append(query_in_views_fov)
            yaw_error_deg_end = final_yaws_deg[-1] if final_yaws_deg.size else np.nan
            end_yaw_error_deg.append(yaw_error_deg_end)

            frac_in_view = query_in_views_fov.mean() if query_in_views_fov.size else np.nan

            symprog = (dist_initial - dist_to_boundary)/(dist_initial+dist_to_boundary)

            print(f"Trajectory {idx}:  distance to boundary = {dist_to_boundary:.3f}m")
            print(f"Trajectory {idx}:  distance to outer boundary = {dist_to_outer_boundary:.3f}m")
            print(f"Trajectory {idx}:  symmetric progress = {symprog:.3f}")
            print(f"Trajectory {idx}:  yaw error = {yaw_error_deg_end:.1f}°")
            print(f"Trajectory {idx}:  query-in view fraction = {frac_in_view:.6f}")

    else:
        # Compute and print the normalized distance to the boundary of the 1.5m circle for all trajectories
        for idx, Xact in enumerate(XXact):
            if Xact.shape[1] == 0:
                continue
            goal_pos = np.array(query_location[:2])
            final_pos_xy = Xact[0:2, -1]
            final_dist_to_center = np.linalg.norm(final_pos_xy - goal_pos)

            if final_dist_to_center < goal_radius:
                traj_xy = Xact[0:2, :]
                dists_to_center = np.linalg.norm(traj_xy - goal_pos[:, None], axis=0)
                boundary_idx = np.argmin(np.abs(dists_to_center - goal_radius))
                eval_pos_xy = traj_xy[:, boundary_idx]
                dist_to_center = dists_to_center[boundary_idx]
                final_quat = Xact[6:10, boundary_idx]
                idx_end = boundary_idx
            else:
                eval_pos_xy = final_pos_xy
                dist_to_center = final_dist_to_center
                final_quat = Xact[6:10, -1]
                idx_end = Xact.shape[1] - 1

            dist_initial = np.linalg.norm(Xact[0:2, 0] - goal_pos)
            dist_to_boundary = abs(dist_to_center - goal_radius)
            dist_to_outer_boundary = abs(dist_to_center - goal_radius_1)

            # Compute yaw error arrays throughout the trajectory up until idx_end
            final_yaws_deg = []
            query_in_views_fov = []
            traj_xy = Xact[0:2, :]
            for t in range(idx_end + 1):
                quat_t = Xact[6:10, t]
                qx, qy, qz, qw = quat_t
                actual_yaw_t = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
                pos_t = traj_xy[:, t]
                goal_direction_t = goal_pos - pos_t
                if np.linalg.norm(goal_direction_t) > 1e-6:
                    required_yaw_t = np.arctan2(goal_direction_t[1], goal_direction_t[0])
                    yaw_error_t = abs(actual_yaw_t - required_yaw_t)
                    if yaw_error_t > np.pi:
                        yaw_error_t = 2 * np.pi - yaw_error_t
                    yaw_err_deg = np.degrees(yaw_error_t)
                else:
                    yaw_err_deg = 0.0
                final_yaws_deg.append(yaw_err_deg)
                query_in_views_fov.append(abs(yaw_err_deg) <= fov_half_deg)

            final_yaws_deg = np.array(final_yaws_deg, dtype=float)
            query_in_views_fov = np.array(query_in_views_fov, dtype=bool)

            all_final_yaws_deg.append(final_yaws_deg)
            all_query_in_views_fov.append(query_in_views_fov)
            yaw_error_deg_end = final_yaws_deg[-1] if final_yaws_deg.size else np.nan
            end_yaw_error_deg.append(yaw_error_deg_end)

            frac_in_view = query_in_views_fov.mean() if query_in_views_fov.size else np.nan

            symprog = (dist_initial - dist_to_boundary)/(dist_initial+dist_to_boundary)

            traj_xy = Xact[0:2, :]
            # traj_len = np.sum(np.linalg.norm(traj_xy[:, 1:] - traj_xy[:, :-1], axis=0))
            # normalized_dist = dist_to_boundary / dist_initial if dist_initial > 0 else np.nan

            print(f"Trajectory {idx}:  distance to boundary = {dist_to_boundary:.3f}m")
            print(f"Trajectory {idx}:  distance to outer boundary = {dist_to_outer_boundary:.3f}m")
            print(f"Trajectory {idx}:  symmetric progress = {symprog:.3f}")
            print(f"Trajectory {idx}:  yaw error = {yaw_error_deg_end:.1f}°")
            print(f"Trajectory {idx}:  query-in view fraction = {frac_in_view:.6f}")

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
    
    return all_query_in_views_fov, all_final_yaws_deg, end_yaw_error_deg

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
    
    # Add figure title showing folder name
    fig1.suptitle(f'3D Trajectory Review - Folder: {folder}', fontsize=12, fontweight='bold')

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