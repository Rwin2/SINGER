import os
import sys, importlib, pkgutil
import torch

import numpy as np

import sousvide.control.policies.ComponentNetworks as cn
from sousvide.control.policies.SousVide_v1 import SousVide_v1
from sousvide.control.policies.hpnet import HPNet
from sousvide.control.policies.svnet import SVNet
from sousvide.control.policies.svnet_no_rma import SVNetNoRMA
from sousvide.control.policies.svnet_no_pretrain import SVNetNoPreTrain
from sousvide.control.policies.svnet_direct import SVNetDirect
from sousvide.control.policies.AdvisorNetworks import SimpleAdvisor
from typing import Union,Tuple,Dict,Any


def _alias_controller_to_control():
    """
    Make `import controller...` resolve under `sousvide.control...`
    so that torch.load can unpickle models that used to live in
    controller.policies.* but now sit under sousvide.control.policies.*.
    """
    # 1) Import the real root packages
    root_ctrl = importlib.import_module("sousvide.control")
    root_pols = importlib.import_module("sousvide.control.policies")

    # 2) Alias top‐level names
    sys.modules["controller"] = root_ctrl
    sys.modules["controller.policies"] = root_pols

    # 3) Recursively alias all submodules of sousvide.control.policies
    prefix = "sousvide.control.policies."
    for finder, full_name, ispkg in pkgutil.walk_packages(root_pols.__path__, prefix):
        mod = importlib.import_module(full_name)
        alias = full_name.replace("sousvide.control", "controller", 1)
        sys.modules[alias] = mod

def policy_factory(path:str,config:Dict[str,Any],device:torch.device) -> Tuple[Union[
    HPNet,SVNet,SVNetNoRMA,SVNetNoPreTrain,SVNetNoPreTrain,SVNetDirect],int]:
    """
    Factory function for creating a policy model.

    Args:
        path:   Path to the model.
        config: Configuration dictionary for the model. 

    Returns:y
        model: Policy model.    
    """

    # Some Useful Intermediate Variables
    model_path = os.path.join(path,"model.pth")
    if config.get("retrain",False) and os.path.isfile(model_path):
        import datetime
        import shutil
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_filename = f"model_backup_{ts}.pth"
        backup_path = os.path.join(path, backup_filename)
        os.makedirs(path, exist_ok=True)
        shutil.copy(model_path, backup_path)
        print(f"[policy_factory] Backed up old checkpoint to {backup_filename}")

        losses_cmd = os.path.join(path, "losses_Commander.pt")
        if os.path.isfile(losses_cmd):
            backup_losses = os.path.join(path, f"losses_Commander_backup_{ts}.pt")
            shutil.copy(losses_cmd, backup_losses)
            print(f"[policy_factory] Backed up losses_Commander ➞ {os.path.basename(backup_losses)}")
            os.remove(losses_cmd) # messy implementation, #FIXME later
            print(f"[policy_factory] Zeroed out {os.path.basename(losses_cmd)}")

        policy_type = config["type"]
        # (1) Instantiate a brand‐new policy so that VisionMLP+CommanderSV are random.
        if policy_type == "SVNet":
            new_model = SVNet(config)
        elif policy_type == "SousVide_v1":
            # backward‐compatibility alias
            new_model = SVNet(config)
        else:
            raise ValueError(f"Unknown policy type '{policy_type}'")


        # (2) Load the old, fully‐trained policy into a temporary object
        try:
            old_model = torch.load(model_path, map_location=device)
        except ModuleNotFoundError as e:
            print(f"Old model found, aliasing controller module to control")
            if "No module named 'controller'" in str(e):            
                _alias_controller_to_control()
                old_model = torch.load(model_path, map_location=device)
            else:
                raise
            
        # (3) Copy ONLY the HistoryEncoder weights from old_model → new_model
        #     We assume both classes define `self.network["HistoryEncoder"]` in the same way.
        try:
            hist_old_state = old_model.network["HistoryEncoder"].state_dict()
        except AttributeError:
            raise RuntimeError(
                "Old policy did not have network['HistoryEncoder']"
            )

        # Now load that state dict into new_model.history
        new_model.network["HistoryEncoder"].load_state_dict(hist_old_state)
        torch.save(new_model, model_path)

        # (4) Optionally delete old_model
        del old_model

        return new_model, new_model.Nz

    # Check if model exists, if not create one.
    if os.path.isfile(model_path):
        pass
    else:
        # Create Folder/File
        os.makedirs(path,exist_ok=True)

        policy_type = config["type"]
        if policy_type == "HPNet":
            torch.save(HPNet(config),model_path)
        elif policy_type == "SVNet":
            torch.save(SVNet(config),model_path)
        elif policy_type == "SVNetNoRMA":
            torch.save(SVNetNoRMA(config),model_path)
        elif policy_type == "SVNetNoPreTrain":
            torch.save(SVNetNoPreTrain(config),model_path)
        elif policy_type == "SVNetDirect":
            torch.save(SVNetDirect(config),model_path)
        elif policy_type == "SousVide_v1":
            torch.save(SousVide_v1(config),model_path)
        else:
            raise ValueError(f"Policy '{policy_type}' not found.")

    # Load Model
    try:
        model = torch.load(model_path,map_location=device)
    except ModuleNotFoundError as e:
        print(f"Old model found, aliasing controller module to control")
        if "No module named 'controller'" in str(e):
            _alias_controller_to_control()
            model = torch.load(model_path, map_location=device)
        else:
            raise
        
    # Extract feature vector size
    Nz = model.Nz

    return model,Nz

def advisor_factory(path:str,config:dict,device:torch.device) -> SimpleAdvisor:
    """
    Factory function for creating a policy model.

    Args:
        path:   Path to the model.
        config: Configuration dictionary for the model. 

    Returns:
        model: Policy model.    
    """

    # Some Useful Intermediate Variables
    model_path = os.path.join(path,"advisor.pth")

    # Check if model exists, if not create one.
    if os.path.isfile(model_path):
        pass
    else:
        # Create Folder/File
        os.makedirs(path,exist_ok=True)

        # Create Model
        torch.save(SimpleAdvisor(config),model_path)

    # Load Model
    model = torch.load(model_path,map_location=device)
    
    return model    
