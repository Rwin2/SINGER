from cv2 import mean
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Literal
import os
import torch
import sousvide.visualize.plot_synthesize as ps
from tabulate import tabulate

def plot_losses(cohort_name:str,roster:List[str],network:Literal["Parameter","Odometry","Commander"]):
    """
    Plot the losses for each student in the roster.
    """

    # # Clear all plots
    # plt.close('all')

    # Some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path = os.path.join(workspace_path,"cohorts",cohort_name)

    # Print some overall stuff
    print("=====================================================")
    print("Network Type: ",network)
    print("Cohort:  ",cohort_name)

    # Plot Losses
    fig, axs = plt.subplots(1, 2, figsize=(5, 3))

    labels = []
    for student_name in roster:
        try:
            student_path = os.path.join(cohort_path,"roster",student_name)
            losses_path = os.path.join(student_path,"losses_"+network+".pt")

            losses = torch.load(losses_path)
            labels.append(student_name)
        except:
            print("-----------------------------------------------------")
            print("No",network,"network found for",student_name)
            print("-----------------------------------------------------")
            continue
        
        if "Neps" in losses.keys():
            print("-----------------------------------------------------")
            print("Student:",student_name)
            try:
                print("Epochs :",sum(losses["Neps"]))
            except:
                print("Old Losses File, SFTI Fallback")
                print("Epochs :",losses["Neps"])

            try:
                if len(set(losses["Nspl"])) == 1:
                    print("Samples:", losses["Nspl"][0])
                else:
                    print("Samples:", losses["Nspl"])
            except:
                print("Samples: ",losses["Nspl"])

            try:
                hours = sum(losses["t_train"]) // 3600
                minutes = (sum(losses["t_train"]) % 3600) // 60
                seconds = np.around(sum(losses["t_train"]) % 60,1)
                print(f"t_train: {hours} hour(s), {minutes} minute(s), {seconds} second(s)")
                # print("t_train: ",losses["t_train"])
            except:
                hours = losses["t_train"] // 3600
                minutes = (losses["t_train"] % 3600) // 60
                seconds = losses["t_train"] % 60
                (f"t_train: {hours} hour(s), {minutes} minute(s), {seconds} second(s)")
                print("t_train: ",losses["t_train"])
        else:
            print("-----------------------------------------------------")
            print("Student:",student_name)
            print("No epochs found in losses file.")
            print(f"losses keys: {list(losses.keys())}")
            print("-----------------------------------------------------")
            continue

        
        loss_train = np.hstack(losses["train"])
        try:
            loss_test = np.hstack(losses["test"])
        except:
            loss_test = np.hstack(losses["tests"])
        axs[0].plot(loss_train)
        axs[0].set_title('Training Loss')

        axs[1].plot(loss_test)
        axs[1].set_title('Testing Loss')

    # Set common labels
    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='Loss')

    # Add Legend
    fig.legend(labels, loc='upper right')

    # # Set the y-axis limits
    # axs[0].set_ylim([0, 0.06])
    # axs[1].set_ylim([0, 0.06])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show(block=False)

def review_simulations(cohort_name:str,course_name:str,roster:List[str],plot_show:bool=False):
    """
    Plot the simulations for each student in the roster.

    """
    
    # Add Expert to Roster
    roster = ["expert"]+roster

    # Initialize Table for plotting and visualization
    headers=["Mean Solve (Hz)","Worst Solve (Hz)",
             "Pos TTE (m)","Best Pos TTE (m)"]
    table = []

    # Some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    output_path = os.path.join(workspace_path,"cohorts",cohort_name,"output")

    # Print some overall stuff
    print("========================================================================================================")
    print("Cohort Name:",cohort_name)
    print("Course Name:",course_name)

    print("Roster:",roster)
    for pilot_name in roster:
        # Load Simulation Data
        trajectories = torch.load(os.path.join(output_path,"sim_"+course_name+"_"+pilot_name+".pt"))
        
        Ebnd = np.zeros((len(trajectories),trajectories[0]["Ndata"]))
        Tsol = np.zeros((len(trajectories),trajectories[0]["Ndata"]))
        methods = []
        for idx,trajectory in enumerate(trajectories):
            # Extract Method Name
            method_name = trajectory["rollout_id"].split("_")[0]
            if method_name not in methods:
                methods.append(method_name)

            # Error Bounds
            for i in range(trajectory["Ndata"]):
                Ebnd[idx,i] = np.min(np.linalg.norm(trajectory["Xro"][:,i].reshape(-1,1)-trajectory["tXUd"][1:11,:],axis=0))

            # Total Solve Time
            Tsol[idx,:] = np.sum(trajectory["Tsol"],axis=0)

        # Trajectory Data
        pilot_name = pilot_name
        mean_solve = 1/np.mean(Tsol)
        worst_solve = 1/np.max(Tsol)
        mean_error = np.mean(Ebnd)
        mean_error_traj = np.mean(Ebnd,axis=1)
        best_error_idx = np.argmin(mean_error_traj)
        best_error = mean_error_traj[best_error_idx]

        # Append Data
        table.append([pilot_name,mean_solve,worst_solve,mean_error,best_error])

        if plot_show:
            print("========================================================================================================")
            print("Visualization ------------------------------------------------------------------------------------------")
            print("Pilot Name    :",pilot_name)
            print("Test Method(s):",methods)
            ps.RO_to_spatial(trajectories,plot_last=True,tXUd=trajectories[0]["tXUd"])

    print("========================================================================================================")
    print("Performance --------------------------------------------------------------------------------------------")
    print(tabulate(table,headers=headers,tablefmt="grid"))
