import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import figs.utilities.trajectory_helper as th

from typing import Dict,Union,Tuple,List
from scipy.spatial.transform import Rotation as R
from sousvide.control.pilot import Pilot

def CP_to_spatial(Tp:List[np.ndarray],CP:List[np.ndarray],
                hz:int=20,n:int=500,plot_last:bool=False):

    # Unpack the trajectory
    XX:List[np.ndarray] = []
    Xmax,Ymax = [],[]
    Xmin,Ymin = [],[]
    for i in range(len(Tp)):
        _,X = unpack_trajectory(Tp[i],CP[i],hz)
        XX.append(X)

        Xmax.append(np.max(X[0,:]))
        Ymax.append(np.max(X[1,:]))
        Xmin.append(np.min(X[0,:]))
        Ymin.append(np.min(X[1,:]))
    xmax,xmin = np.ceil(np.max(Xmax)),np.floor(np.min(Xmin))
    ymax,ymin = np.ceil(np.max(Ymax)),np.floor(np.min(Ymin))

    # Initialize World Frame Plot
    traj_colors = ["red","green","blue","orange","purple","brown","pink","gray","olive","cyan"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plim = np.array([
        [ xmin, xmax],
        [ ymin, ymax],
        [  0.0, -3.0]])    

    xlim = plim[0,:]
    ylim = plim[1,:]
    zlim = plim[2,:]
    ratio = plim[:,1]-plim[:,0]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_box_aspect(ratio)  # aspect ratio is 1:1:1 in data space

    ax.invert_zaxis()
    ax.invert_yaxis()

    # Rollout the world frame trajectory
    for idx,X in enumerate(XX):
        # Plot the world frame trajectory
        ax.plot(X[0,:], X[1,:], X[2,:],color=traj_colors[idx%len(traj_colors)],alpha=0.5)             # spline

        for i in range(0,X.shape[1],n):
            quad_frame(X[:,i],ax)

        if plot_last == True or idx == 0:
            quad_frame(X[:,-1],ax)

    # Plot the control points
    for idx,CPk in enumerate(CP):
        for i in range(CPk.shape[0]):
            ax.scatter(CPk[i,0,0],CPk[i,1,0],CPk[i,2,0],color=traj_colors[idx%len(traj_colors)],marker='x')
    plt.show(block=False)

def CP_to_time(Tp:np.ndarray,CP:np.ndarray,hz:int=30):
    
    # Unpack the trajectory
    TT:List[np.ndarray] = []
    XX:List[np.ndarray] = []
    Xmax,Ymax,Xmin,Ymin = [],[],[],[]
    for i in range(len(Tp)):
        T,X = unpack_trajectory(Tp[i],CP[i],hz)
        TT.append(T)
        XX.append(X)

        Xmax.append(np.max(X[0,:]))
        Ymax.append(np.max(X[1,:]))
        Xmin.append(np.min(X[0,:]))
        Ymin.append(np.min(X[1,:]))

    xmax,xmin = np.ceil(np.max(Xmax)),np.floor(np.min(Xmin))
    ymax,ymin = np.ceil(np.max(Ymax)),np.floor(np.min(Ymin))

    xlim = np.array([
        [ xmin, xmax],
        [ ymin, ymax],
        [  1.0, -3.0],
        [ -3.0,  3.0],
        [ -3.0,  3.0],
        [  3.0, -3.0],
        [ -1.2,  1.2],
        [ -1.2,  1.2],
        [ -1.2,  1.2],
        [ -1.2,  1.2],
        [ -5.0,  5.0],
        [ -5.0,  5.0],
        [ -5.0,  5.0]])
    
    # Plot Positions and Velocities
    ylabels = [["$p_x$","$p_y$","$p_z$"],["$v_x$","$v_y$","$v_z$"]]
    fig, axs = plt.subplots(3, 2, figsize=(10, 4))
    for i in range(2):
        for j in range(3):
            idd = j+(3*i)

            for idx,X in enumerate(XX):
                axs[j,i].plot(TT[idx],X[idd,:],alpha=0.5)

            axs[j,i].set_ylim(xlim[idd,:])
            axs[j,i].set_ylabel(ylabels[i][j])

    axs[0, 0].set_title('Position')
    axs[0, 1].set_title('Velocity')
    
    plt.tight_layout()
    plt.show()

    # Plot Orientation and Body Rates
    ylabels = [["$q_x$","$q_y$","$q_z$","q_w"],[r"$\omega_x$",r"$\omega_y$",r"$\omega_z$"]]
    fig, axs = plt.subplots(4, 2, figsize=(10, 4))
    for i in range(2):
        for j in range(4):
            if not ((i==1) and (j==3)):
                idd = 6+j+(4*i)

                for idx,X in enumerate(XX):
                    axs[j,i].plot(TT[idx],X[idd,:],alpha=0.5)
                
                axs[j,i].set_ylim(xlim[idd,:])
                axs[j,i].set_ylabel(ylabels[i][j])

    axs[0, 0].set_title('Orientation')
    axs[0, 1].set_title('Body Rates')
    
    plt.tight_layout()
    plt.show(block=False)

def CP_to_fo(Tp:np.ndarray,CP:np.ndarray,hz:int=30):
    # # Clear all plots
    # plt.close('all')

    # Unpack the trajectory
    TT:List[np.ndarray] = []
    XX:List[np.ndarray] = []
    for i in range(len(Tp)):
        T,X = unpack_trajectory(Tp[i],CP[i],hz=hz,mode='fo')
        TT.append(T)
        XX.append(X)

    # Plot Position and Velocities
    ylabels = [["$p_x$","$p_y$","$p_z$", r"$\psi$"],["$v_x$","$v_y$","$v_z$", r"$\dot{\psi}$"]]
    fig, axs = plt.subplots(4, 2, figsize=(10, 4))
    for i in range(4):
        for j in range(2):
            for idx,X in enumerate(XX):
                axs[i,j].plot(TT[idx],X[:,i,j],alpha=0.5)
                axs[i,j].set_ylabel(ylabels[j][i])

    axs[0, 0].set_title('Position')
    axs[0, 1].set_title('Velocity')
    
    plt.tight_layout()
    plt.show(block=False)

    # Plot Accelerations and Jerks
    ylabels = [["$a_x$","$a_y$","$a_z$", r"$\ddot{\psi}$"],["$j_x$","$j_y$","$j_z$", r"$\dddot{\psi}$"]]
    fig, axs = plt.subplots(4, 2, figsize=(10, 4))
    for i in range(4):
        for j in range(2):
            for idx,X in enumerate(XX):
                axs[i,j].plot(TT[idx],X[:,i,j+2],alpha=0.5)
                axs[i,j].set_ylabel(ylabels[j][i])

    axs[0, 0].set_title('Acceleration')
    axs[0, 1].set_title('Jerk')
    
    plt.tight_layout()
    plt.show()

def tXU_to_spatial(tXU_list:List[np.ndarray],
              n:int=None):
    
    # Initialize World Frame Plot
    traj_colors = ["red","green","blue","orange","purple","brown","pink","gray","olive","cyan"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plim = np.array([
        [ -8.0,  8.0],
        [ -8.0,  8.0],
        [  1.0, -3.0]])
    
    xlim = plim[0,:]
    ylim = plim[1,:]
    zlim = plim[2,:]
    ratio = plim[:,1]-plim[:,0]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_box_aspect(ratio)  # aspect ratio is 1:1:1 in data space

    ax.invert_zaxis()
    ax.invert_yaxis()

    # Rollout the world frame trajectory
    for idx,tXU in enumerate(tXU_list):
        # Plot the world frame trajectory
        ax.plot(tXU[1,:], tXU[2,:], tXU[3,:],color=traj_colors[idx%len(traj_colors)],alpha=0.5)             # spline

        # Plot Initial and Final
        quad_frame(tXU[1:14,0],ax)
        quad_frame(tXU[1:14,-1],ax)
            
        if n is not None:
            for i in range(n,tXU.shape[1],n):
                quad_frame(tXU[1:14,i],ax)


    plt.show(block=False)

def RO_to_spatial(RO:List[Dict[str,Union[np.ndarray,int]]],
              n:int=None,scale=1.0,plot_last:bool=False,
              tXUd:Union[None,np.ndarray]=None):

    # Initialize World Frame Plot
    traj_colors = ["red","green","blue","orange","purple","brown","pink","gray","olive","cyan"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    Xmax,Ymax,Xmin,Ymin = [],[],[],[]
    for idx,ro in enumerate(RO):
        Xro = ro["Xro"]

        Xmax.append(np.max(Xro[0,:]))
        Ymax.append(np.max(Xro[1,:]))
        Xmin.append(np.min(Xro[0,:]))
        Ymin.append(np.min(Xro[1,:]))

    xmax,xmin = np.ceil(np.max(Xmax)),np.floor(np.min(Xmin))
    ymax,ymin = np.ceil(np.max(Ymax)),np.floor(np.min(Ymin))
    plim = np.array([
        [ xmin,  xmax],
        [ ymin,  ymax],
        [  1.0, -3.0]])
    
    xlim = plim[0,:]
    ylim = plim[1,:]
    zlim = plim[2,:]
    ratio = plim[:,1]-plim[:,0]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_box_aspect(ratio)  # aspect ratio is 1:1:1 in data space

    ax.invert_zaxis()
    ax.invert_yaxis()

    # Rollout the world frame trajectory
    for idx,ro in enumerate(RO):
        Xro = ro["Xro"]

        # Plot the world frame trajectory
        ax.plot(Xro[0,:], Xro[1,:], Xro[2,:],color=traj_colors[idx%len(traj_colors)],alpha=0.5)             # spline

        # Plot Initial and Final
        quad_frame(Xro[:,0],ax,scale=scale)

        if plot_last == True:
            quad_frame(Xro[:,-1],ax,scale=scale)
            
        if n is not None:
            for i in range(n,Xro.shape[1],n):
                quad_frame(Xro[:,i],ax,scale=scale)

    if tXUd is not None:
        ax.plot(tXUd[1,:], tXUd[2,:], tXUd[3,:],color='k', linestyle='--',linewidth=0.8)             # spline
    
    
    ref, = ax.plot([], [], 'k--', label='reference')
    fig.legend(handles=[ref],loc='upper right', bbox_to_anchor=(0.9, 0.7, 0.0, 0.1))
    
    plt.tight_layout()
    plt.show(block=False)

def RO_to_time(RO:List[Dict[str,Union[np.ndarray,int]]],tXUd:Union[None,np.ndarray]=None):
    # # State plot limits
    # plim = np.array([
    #     [ -6.0,  6.0],
    #     [ -6.0,  6.0],
    #     [  1.0, -5.0]])
    # vlim = np.array([
    #     [ -3.0,  3.0],
    #     [ -3.0,  3.0],
    #     [  3.0, -3.0]])
    # qlim = np.array([
    #     [ -1.2,  1.2],
    #     [ -1.2,  1.2],
    #     [ -1.2,  1.2],
    #     [ -1.2,  1.2]])
    
    # Plot Positions and Velocities
    ylabels = [["$p_x$","$p_y$","$p_z$"],["$v_x$","$v_y$","$v_z$"]]
    fig, axs = plt.subplots(3, 2, figsize=(10, 4))
    for i in range(2):
        for j in range(3):
            idd = j+(3*i)
            for ro in RO:
                Tro,Xro = ro["Tro"],ro["Xro"]  
                axs[j,i].plot(Tro,Xro[idd,:],alpha=0.5)

            if tXUd is not None:
                axs[j,i].plot(tXUd[0,:],tXUd[1+idd,:],color='k', linestyle='--',linewidth=0.8)
        
            axs[j,i].set_ylabel(ylabels[i][j])

            if tXUd is not None:
                axs[j,i].set_xlim([0.0,tXUd[0,-1]])
            # if i == 0:
            #     axs[j,i].set_ylim(plim[j,:])
            # else:
            #     axs[j,i].set_ylim(vlim[j,:])

    axs[0, 0].set_title('Position')
    axs[0, 1].set_title('Velocity')
    
    ref, = axs[0, 0].plot([], [], 'k--', label='reference')
    fig.legend(handles=[ref],loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Adjust the subplot layout to make room for the legend
    plt.show()

    # Plot Orientation and Control Inputs
    ylabels = [[r"$q_x$", r"$q_y$", r"$q_z$", "q_w"], [r"$f_{th}$", r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]]
    fig, axs = plt.subplots(4, 2, figsize=(10, 4))
    for i in range(4):
        idd = 6+i

        for ro in RO:
            Tro,Xro = ro["Tro"],ro["Xro"]
            axs[i,0].plot(Tro,Xro[idd,:],alpha=0.5)

        if tXUd is not None:
            axs[i,0].plot(tXUd[0,:],tXUd[7+i,:],color='k', linestyle='--',linewidth=0.8)

        axs[i,0].set_ylabel(ylabels[0][i])
        if tXUd is not None:
            axs[i,0].set_xlim([0.0,tXUd[0,-1]])
        # axs[j].set_ylim(qlim[j,:])

    for i in range(4):
        for ro in RO:
            Tro,Uro = ro["Tro"],ro["Uro"]
            Uro[0,:] = Uro[0,:]

            axs[i,1].plot(Tro[0:-1],Uro[i,:],alpha=0.5)

        if tXUd is not None:
            axs[i,1].plot(tXUd[0,:],tXUd[11+i,:],color='k', linestyle='--',linewidth=0.8)
        
        axs[i,1].set_ylabel(ylabels[1][i])

        if tXUd is not None:
            axs[i,1].set_xlim([0.0,tXUd[0,-1]])

    axs[0, 1].invert_yaxis()
    axs[0,0].set_title('Orientation')
    axs[0,1].set_title('Control Inputs')

    ref, = axs[0, 0].plot([], [], 'k--', label='reference')
    fig.legend(handles=[ref],loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Adjust the subplot layout to make room for the legend
    plt.show(block=False)

def tXU_to_time(tXU_list:List[np.ndarray],q_clean:bool=True):
    # # Clear all plots
    # plt.close('all')

    # Clean Quaternions
    if q_clean == True:
        for tXU in tXU_list:
            for i in range(1,tXU.shape[1]):
                tXU[7:11,i] = th.obedient_quaternion(tXU[7:11,i],tXU[7:11,i-1])

    # Plot Positions and Velocities
    ylabels = [["$p_x$","$p_y$","$p_z$"],["$v_x$","$v_y$","$v_z$"]]
    fig, axs = plt.subplots(3, 2, figsize=(10, 4))
    for i in range(2):
        for j in range(3):
            idd = j+(3*i)

            for tXU in tXU_list:
                T = tXU[0,:]
                X = tXU[1:14,:]

                axs[j,i].plot(T,X[idd,:],alpha=0.5)
                axs[j,i].set_ylabel(ylabels[i][j])

                if i == 0 and j == 0:
                    axs[j,i].set_ylim([-7.0,7.0])
                else:
                    axs[j,i].set_ylim([-3.0,3.0])

    axs[0, 0].set_title('Position')
    axs[0, 1].set_title('Velocity')
    
    plt.tight_layout()
    plt.show(block=False)

    # Plot Orientation and Body Rates
    ylabels = [["$q_x$","$q_y$","$q_z$","$q_w$"],[r"$\omega_x$",r"$\omega_y$",r"$\omega_z$"]]
    fig, axs = plt.subplots(4, 2, figsize=(10, 4))
    for i in range(2):
        for j in range(4):
            if not ((i==1) and (j==3)):
                idd = 6+j+(4*i)

                for tXU in tXU_list:
                    T = tXU[0,:]
                    X = tXU[1:14,:]

                    axs[j,i].plot(T,X[idd,:],alpha=0.5)
                    axs[j,i].set_ylabel(ylabels[i][j])
                    axs[j,i].set_ylim([-1.5,1.5])

    axs[0, 0].set_title('Orientation')
    axs[0, 1].set_title('Body Rates')
    
    plt.tight_layout()
    plt.show(block=False)

def plot_Ubr(tXU_list:List[np.ndarray]):
    # # Clear all plots
    # plt.close('all')

    # Plot the Control Inputs
    ylabels = ["$f$","$wx$","$wy$","$wz$"]
    fig, axs = plt.subplots(4, 1, figsize=(10, 4))
    for i in range(4):
        for tXU in tXU_list:
            U = tXU[14:18,:]

            axs[i].plot(U[i,:],alpha=0.5)
            axs[i].set_ylabel(ylabels[i])
            # axs[i].set_ylim([-5.0,5.0])

    axs[0].set_title('Control Inputs')
    
    plt.tight_layout()
    plt.show(block=False)

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

def unpack_trajectory(Tp:np.ndarray,CP:np.ndarray,hz:int,
                      mode:str='time',trim:bool=True) -> Tuple[np.ndarray,np.ndarray]:
    """
    Unpack the trajectory from the control points.
    """

    # Unpack the trajectory
    Nt = int(Tp[-1]*hz+1)
    T = np.zeros(Nt)

    if mode == 'time':
        X = np.zeros((13,Nt))
    else:
        X = np.zeros((Nt,4,12))
        
    idx = 0
    for k in range(Nt):
        tk = Tp[0]+k/hz

        if trim == True:
            tk = np.minimum(Tp[-1],tk)
        if tk > Tp[idx+1] and idx < len(Tp)-2:
            idx += 1

        t0,tf = Tp[idx],Tp[idx+1]
        fo = th.ts_to_fo(tk-t0,tf-t0,CP[idx,:,:])

        T[k] = tk
        if mode == 'time':
            X[:,k] = th.fo_to_xu(fo)[0:13]
        else:
            X[k,:,:] = fo

    # Ensure continuity of the quaternion
    if mode == 'time':
        qr = np.array([0.0,0.0,0.0,1.0])
        for k in range(Nt):
            q = X[6:10,k]
            qc = th.obedient_quaternion(q,qr)

            X[6:10,k] = qc
            qr = qc
        
    return T,X

def plot_rollout_data(cohort:str,Nsamples:int=50,random:bool=True):
    """"
    Plot the rollout data for a cohort.
    """

    # Generate some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path = os.path.join(workspace_path,"cohorts",cohort)
    rollout_path = os.path.join(cohort_path,"rollout_data")

    # Load Flight Data
    for subdir in os.listdir(rollout_path):
        subdir_path = os.path.join(rollout_path, subdir)
        
        # Check if it is a directory
        if os.path.isdir(subdir_path):
            # List all files in the subdirectory
            files = os.listdir(subdir_path)
            
            # Filter out only .pt files
            files = [f for f in os.listdir(subdir_path) if f.startswith("trajectories")]

            # If there are no .pkl files in this subdirectory, skip it
            if not files:
                continue
            
            # Select a .pt file
            if random == True:
                traj_file = np.random.choice(files)
            else:
                traj_file = files[0]
            traj_path = os.path.join(subdir_path, traj_file)
            
            # Load the trajectories
            trajectories = torch.load(traj_path)

            # Trim the number of samples
            if Nsamples > len(trajectories['data']):
                Nsamples = len(trajectories['data'])
                print(f"Only {Nsamples} samples available in {traj_file}. Showing all samples.")
            else:
                print(f"Showing {Nsamples} samples from {traj_file}")
            trajectories['data'] = trajectories['data'][0:Nsamples]

            # Plot the data
            RO_to_spatial(trajectories['data'],scale=0.5,tXUd=trajectories['tXUd'])
            RO_to_time(trajectories['data'],tXUd=trajectories['tXUd'])

def plot_observation_data(cohort:str,roster:List[str],random:bool=True):
    """"
    Plot the observation data for a cohort.
    """

    # Generate some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path = os.path.join(workspace_path,"cohorts",cohort)
    observation_data_path = os.path.join(cohort_path,"observation_data")

    print("==========================================================================================")
    print("Observation Data Summary")
    print("==========================================================================================")

    print("Cohort Name :",cohort)

    # Review the observation data
    for pilot_name in roster:
        # Get number of observation files
        observation_files = []
        pilot_path = os.path.join(observation_data_path,pilot_name)
        for root, _, files in os.walk(pilot_path):
            for file in files:
                observation_files.append(os.path.join(root, file))

        Nobsf = len(observation_files)

        # Get some data insights
        observation_files = sorted(observation_files)

        # Get approximate number of observations
        observations = torch.load(observation_files[0])
        Ndata = (Nobsf-1)*observations['Nobs']
        
        # Load pilot
        pilot = Pilot(cohort,pilot_name)

        print("------------------------------------------------------------------------------------------")
        print(f"Pilot Name        : {pilot.name}")
        print(f"Neural Network(s) : {pilot.model.name} [{', '.join(pilot.model.network.keys())}]")
        print(f"Approx. Data Count: {Ndata}")

    print("==========================================================================================")
