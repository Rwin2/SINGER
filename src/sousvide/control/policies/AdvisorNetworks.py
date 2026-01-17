import torch
from torch import nn
from sousvide.control.policies.BaseNetworks import *
from typing import List

class SimpleAdvisor(nn.Module):
    def __init__(self,cfg_model:str):
        """
        Defines a pilot's neural network model. The model is a fully connected
        neural network with ReLU activation functions in the intermediate layers
        and a linear output layer.

        Args:
            cfg_model: Configuration dictionary.

        Variables:
            Nadv: Number of adviser networks.
            networks: List of neural networks.
        """
        # Some Useful Intermediate Variables
        Na_in = len(cfg_model["commander"]["state"])+len(cfg_model["commander"]["objective"])
        Na_out = cfg_model["commander"]["output"]

        # Initial Parent Call
        super(SimpleAdvisor,self).__init__()

        # Adviser networks
        if cfg_model["advisor"]["type"] == None:
            self.Nadv = None
            self.networks = None
        else:
            self.Nadv = cfg_model["advisor"]["count"]
            self.networks = nn.ModuleList([SimpleMLP(Na_in, cfg_model["advisor"]["sizes"], Na_out) for _ in range(self.Nadv)])

    def forward(self,
                dxt:torch.Tensor,upr:torch.Tensor,
                xcr:torch.Tensor,obj:torch.Tensor,
                img:torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network model.

        Args:
            dxt: History inputs.
            upr: Horizon inputs.
            xcr: Course inputs.
            Obj: Objective inputs.
            img: Image inputs.

        Returns:
            y_adv: Adviser networks outputs.
        """
        # Some Useful Intermediate Variables
        _ = img.flatten()                   # Image data is not used in this model
        xhy = torch.cat((dxt,upr),dim=-2).flatten(-2)
        x = torch.cat((xhy,xcr,obj.flatten(1)),-1)                         # Concatenate inputs
        
        # Oracle networks
        if self.Nadv is not None:
            y_adv = []                              
            for network in self.networks[2:]:
                y_adv.append(network(x))
        else:
            y_adv = None

        return y_adv