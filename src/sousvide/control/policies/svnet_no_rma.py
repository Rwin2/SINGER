import torch
from torch import nn
from sousvide.control.policies.ComponentNetworks import *
from typing import Dict,Tuple,Any,Callable

class SVNetNoRMA(nn.Module):    
    def __init__(self,
                 config:Dict[str,Any],
                 name:str="SVNetNoRMA"):
        
        """
        Defines a pilot's neural network model.

        Network Description:
        SV-Net with no RMA. This policies comprises of just the feature network and
        the command network which we train together (under Commander).

        Args:
            config:         Configuration dictionary for the model.
            name:           Name of the policy.

        Variables:
            name:           Name of the policy.
            Nz:             Feature vector size.
            network:        Dictionary of the network components.
            get_data:       Dictionary of functions to extract data for the network.
            get_network:    Dictionary of network components for training.
        """

        # Initial Parent Call
        super(SVNetNoRMA,self).__init__()

        # ----------------------------------------------------------------------------------------------
        # Class Intermediate Variables
        # ----------------------------------------------------------------------------------------------

        vision_mlp_cfg = config["networks"][0]
        command_df_cfg = config["networks"][1]

        vision_mlp_network = VisionMLP(
                vision_mlp_cfg["state"],
                vision_mlp_cfg["state_layer"],
                vision_mlp_cfg["hidden_sizes"],
                vision_mlp_cfg["output_size"],
                vision_mlp_cfg["CNN"]
            )
        commander_df_network = CommandSVNoRMA(
                command_df_cfg["state"],
                command_df_cfg["objective"],
                vision_mlp_cfg["output_size"],
                command_df_cfg["hidden_sizes"],
                command_df_cfg["output_size"]
            )
        
        # ----------------------------------------------------------------------------------------------
        # Class Name Variables
        # ----------------------------------------------------------------------------------------------

        # Network ID
        self.name = name
        self.Nz = 0

        # Network Components           
        self.network = nn.ModuleDict({
            "VisionMLP": vision_mlp_network,
            "CommanderDF": commander_df_network
        })

        # Network Callers [Parameter,Odometry,Commander]
        self.get_data:Dict[str,Callable] = {                         
            "Commander": self.get_commander_data,
            "Images": self.get_image_data
        }

        self.get_network:Dict[str,Dict[str,nn.Module]] = {
            "Commander": {
                "Train" :self,
                "Unlock":self
            }
        }

    def extract_inputs(self,tx:torch.Tensor,Obj:torch.Tensor,
                       Img:torch.Tensor,DxU:torch.Tensor,Znn:torch.Tensor,hy_idx:int) -> Dict[str,torch.Tensor]:
        """
        Takes the basis input variables and extracts the relevant inputs for the model. If the
        model component does not exist, the input is set to None.

        Args:
            tx:     Current state.
            Obj:    Objective.
            Img:    Image.
            DxU:    History deltas.
            Vio:    Odometry.
            hy_idx: History indices.

        Returns:
            xnn:    Extracted inputs for the model

        """

        # Unused Variables
        _ = Znn,DxU,hy_idx

        # Extract Indices
        ix_tx_vis  = self.network["VisionMLP"].ix_tx
        ix_tx_com  = self.network["CommanderDF"].ix_tx
        ix_obj_com = self.network["CommanderDF"].ix_obj

        # Extract Data
        xnn = {
            "tx_com" : tx[ix_tx_com].flatten(),
            "obj_com": Obj[ix_obj_com].flatten(),
            "img_vis": Img.clone(),
            "tx_vis" : tx[ix_tx_vis].flatten()
        }

        return xnn

    def get_commander_inputs(self,xnn: Dict[str,torch.Tensor]) -> Tuple[torch.Tensor,...]:
        return (xnn["tx_com"],xnn["obj_com"],xnn["img_vis"],xnn["tx_vis"])
    
    def get_commander_data(self,xnn: Dict[str,torch.Tensor],ynn: Dict[str,torch.Tensor]) -> Tuple[Tuple[torch.Tensor,...],torch.Tensor]:
        return self.get_commander_inputs(xnn),ynn["unn"]

    def get_image_data(self,xnn: Dict[str,torch.Tensor]) -> torch.Tensor:
        return xnn["img_vis"]
    
    def forward(self,tx_com:torch.Tensor,obj_com:torch.Tensor,
                img_vis:torch.Tensor,tx_vis:torch.Tensor) -> Tuple[torch.Tensor,None]:
        """
        Defines the forward pass of the neural network model.

        Args:
            tx_com:     Current state for commander network.
            obj_com:    Objective for commander network.
            img_vis:    Image for vision network.
            tx_vis:     Current state for vision network.

        Returns:
            ycm: Command network output.
            zcm: Empty.
        """

        # Command Network
        y_vis,_ = self.network["VisionMLP"](img_vis,tx_vis)
        y_com,_ = self.network["CommanderDF"](tx_com,obj_com,y_vis)

        return y_com,None