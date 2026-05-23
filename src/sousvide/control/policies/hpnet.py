import torch
from torch import nn
from sousvide.control.policies.ComponentNetworks import HistoryEncoder,CommandHP
from typing import Dict,Tuple,Any,Callable

class HPNet(nn.Module):
    def __init__(self,
                 config:Dict[str,Any],
                 name:str="HPNet"):
        """
        Defines a pilot's neural network model.

        Network Description:
        HotPot-Net is SV-Net without vision. This serves as a sanity check for learned
        policies by swapping out the image input with a full state input. The policy is
        trained in a manner similar to SV-Net where the history network is pre-trained
        (under Parameter) and then locked during training of the command network (under
        Commander).

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
        super(HPNet,self).__init__()

        # ----------------------------------------------------------------------------------------------
        # Class Intermediate Variables
        # ----------------------------------------------------------------------------------------------

        # Network Configs
        history_enc_cfg = config["networks"][0]
        command_hp_cfg = config["networks"][1]

        history_enc_network = HistoryEncoder(
                history_enc_cfg["delta"],
                history_enc_cfg["frames"],
                history_enc_cfg["hidden_sizes"],
                history_enc_cfg["encoder_size"],
                history_enc_cfg["decoder_size"]
            )
        command_hp_network = CommandHP(
                command_hp_cfg["state"],
                command_hp_cfg["objective"],
                history_enc_cfg["encoder_size"],
                command_hp_cfg["hidden_sizes"],
                command_hp_cfg["output_size"],
            )
            
        # ----------------------------------------------------------------------------------------------
        # Class Name Variables
        # ----------------------------------------------------------------------------------------------

        # Network ID
        self.name = name                            
        self.Nz = 0 

        # Network Components           
        self.network = nn.ModuleDict({              
            "HistoryEncoder": history_enc_network,
            "CommandHP": command_hp_network
        })

        # Network Callers [Parameter,Odometry,Commander]
        self.get_data:Dict[str,Callable] = {                         
            "Parameter": self.get_parameter_data,
            "Commander": self.get_commander_data
        }

        self.get_network:Dict[str,Dict[str,nn.Module]] = {
            "Parameter": {
                "Train" : self.network["HistoryEncoder"],
                "Unlock": self.network["HistoryEncoder"]
            },
            "Commander": {
                "Train" :self,
                "Unlock":self.network["CommandHP"]
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
        _ = Img,Znn

        # Extract Indices
        ix_tx_com  = self.network["CommandHP"].ix_tx
        ix_obj_com = self.network["CommandHP"].ix_obj
        ix_DxU_par = self.network["HistoryEncoder"].ix_DxU

        # xnn Variables
        xnn = {
            "tx_com" : tx[ix_tx_com].flatten(),
            "obj_com": Obj[ix_obj_com].flatten(),
            "dxu_par": DxU[ix_DxU_par[0],hy_idx+ix_DxU_par[1]].flatten()
        }

        return xnn

    def get_commander_inputs(self,xnn: Dict[str,torch.Tensor]) -> Tuple[torch.Tensor,...]:
        return (xnn["tx_com"],xnn["obj_com"],xnn["dxu_par"])
    
    def get_parameter_data(self,xnn:Dict[str,torch.Tensor],ynn:Dict[str,torch.Tensor]) -> Tuple[Tuple[torch.Tensor],torch.Tensor]:
        return (xnn["dxu_par"],),ynn["mfn"]
    
    def get_commander_data(self,xnn: Dict[str,torch.Tensor],ynn: Dict[str,torch.Tensor]) -> Tuple[Tuple[torch.Tensor,...],torch.Tensor]:
        return self.get_commander_inputs(xnn),ynn["unn"]

    def forward(self,tx_com:torch.Tensor,obj_com:torch.Tensor,
                dxu_par:torch.Tensor) -> Tuple[torch.Tensor,None]:
        """
        Defines the forward pass of the neural network model.

        Args:
            tx_com:     Current state for commander network.
            obj_com:    Objective for commander network
            dxu_par:    History deltas for parameter network.

        Returns:
            ycm: Command network output.
            zcm: Empty.
        """

        # Parameter Network
        _,z_par = self.network["HistoryEncoder"](dxu_par)

        # Commander Network
        y_com,_ = self.network["CommandHP"](tx_com,obj_com,z_par)

        return y_com,None