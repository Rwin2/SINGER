import torch
from torch import nn
import numpy as np
from typing import List,Tuple
import sousvide.control.policies.BaseNetworks as bn

class HistoryEncoder(nn.Module):
    def __init__(self,
                 delta:List[int],frames:List[int],
                 hidden_sizes:List[int],
                 encoder_size:int, decoder_size:int):
        """
        Initialize a history MLP model.

        Args:
            delta:          List of delta indices.
            frames:         List of frame indices.
            hidden_sizes:   List of hidden layer sizes.
            encoder_size:   Encoder output size.
            decoder_size:   Decoder output size.

        Variables:
            network:        Neural network.
        """

        # Initialize the parent class
        super(HistoryEncoder, self).__init__()

        # Some useful intermediate variables
        input_size = len(delta)*len(frames)

        # Define the model
        self.ix_DxU = np.ix_(np.array(delta),-1-np.array(frames))
        self.network = bn.SimpleEncoder(input_size, hidden_sizes, encoder_size, decoder_size)
    
    def forward(self, dxu:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            dxu:  History delta tensor.

        Returns:
            ynn:  Decoder output tensor.
            znn:  Encoder output tensor.
        """

        # History MLP
        ynn,znn = self.network(dxu)

        return ynn,znn

class DirectHistoryEncoder(nn.Module):
    def __init__(self, 
                 delta:List[int],frames:List[int],
                 hidden_sizes:List[int],
                 output_size:int):
        """
        Initialize a history MLP model.

        Args:
            delta:          List of delta indices.
            frames:         List of frame indices.
            hidden_sizes:   List of hidden layer sizes.
            output_size:    Encoder output

        Variables:
            networks:       List of neural networks.
        """

        # Initialize the parent class
        super(DirectHistoryEncoder, self).__init__()

        # Some useful intermediate variables
        input_size = len(delta)*len(frames)

        # Define the model
        self.ix_DxU = np.ix_(np.array(delta),-1-np.array(frames))
        self.network = bn.DirectEncoder(input_size, hidden_sizes, output_size)
    
    def forward(self, dxu:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            dxu:  History delta tensor.

        Returns:
            ynn:  Decoder output tensor.
            znn:  Encoder output tensor.
        """

        # History MLP
        ynn,znn = self.network(dxu)

        return ynn,znn
    
class CommandHP(nn.Module):
    def __init__(self, 
                 state:List[int],
                 objective:List[int],
                 fparam_size:int,
                 hidden_sizes:List[int],
                 output_size:int,
                 active_end:bool=False,dropout:float=0.2):
        """
        Initialize a command MLP model.

        Args:
            state:          List of state indices.
            objective:      List of objective indices.
            fparam_size:    Size of the parameter feature vector.
            hidden_sizes:   List of hidden layer sizes.
            output_size:    Output size.
            active_end:     Activation function at the end.
            dropout:        Dropout rate.

        Variables:
            network:       List of neural networks.
        """

        # Initialize the parent class
        super(CommandHP, self).__init__()

        # Some useful intermediate variables
        input_size = len(state) + len(objective) + fparam_size

        # Define the model
        self.ix_tx = np.array(state)
        self.ix_obj = np.array(objective)
        self.network = bn.SimpleMLP(input_size, hidden_sizes, output_size,
                                    active_end=active_end,dropout=dropout)
        
    def forward(self,tx:torch.Tensor,obj:torch.Tensor,zpar:torch.Tensor) -> Tuple[torch.Tensor,None]:
        """
        Forward pass of the model.

        Args:
            tx:     Time+State tensor.
            obj:    Objective tensor.
            zpar:   Parameter tensor.

        Returns:
            ycm:    Output tensor.
            zcm:    None.
        """
        xcm = torch.cat((tx,obj,zpar),-1)
        ycm = self.network(xcm)

        return ycm,None

class CommandSVNoRMA(nn.Module):
    def __init__(self,
                 state:List[int],
                 objective:List[int],
                 fimage_size:int,
                 hidden_sizes:List[int],
                 output_size:int,
                 active_end:bool=False,dropout:float=0.2):
        """
        Initialize a command MLP model.

        Args:
            state:          List of state indices.
            objective:      List of objective indices.
            fimage_size:    Size of the image feature vector.
            hidden_sizes:   List of hidden layer sizes.
            output_size:    Output size.
            active_end:     Activation function at the end.
            dropout:        Dropout rate

        Variables:
            network:       List of neural networks.
        """

        # Initialize the parent class
        super(CommandSVNoRMA, self).__init__()

        # Some useful intermediate variables
        input_size = len(state) + len(objective) + fimage_size

        # Define the model
        self.ix_tx = np.array(state)
        self.ix_obj = np.array(objective)
        self.network = bn.SimpleMLP(input_size, hidden_sizes, output_size,
                                    active_end=active_end,dropout=dropout)

    def forward(self, tx:torch.Tensor,obj:torch.Tensor,zimg:torch.Tensor,) -> Tuple[torch.Tensor,None]:
        """
        Forward pass of the model.

        Args:
            tx:     Time+State tensor.
            obj:    Objective tensor.
            zimg:   Image tensor.

        Returns:
            ycm:  Output tensor.
            zcm:  None.
        """

        # Command MLP
        xcm = torch.cat((tx,obj,zimg),-1)
        ycm = self.network(xcm)

        return ycm,None
    
class CommandSV(nn.Module):
    def __init__(self,
                 state:List[int],
                 objective:List[int],
                 fparam_size:int,
                 fimage_size:int,
                 hidden_sizes:List[int],
                 output_size:int,
                 active_end:bool=False,dropout:float=0.2):
        """
        Initialize a command MLP model.

        Args:
            state:          List of state indices.
            objective:      List of objective indices.
            fparam_size:    Size of the parameter feature vector.
            fimage_size:    Size of the image feature vector.
            hidden_sizes:   List of hidden layer sizes.
            output_size:    Output size.
            active_end:     Activation function at the end.
            dropout:        Dropout rate

        Variables:
            network:       List of neural networks.
        """

        # Initialize the parent class
        super(CommandSV, self).__init__()

        # Some useful intermediate variables
        input_size = len(state) + len(objective) + fimage_size + fparam_size

        # Define the model
        self.ix_tx = np.array(state)
        self.ix_obj = np.array(objective)
        self.network = bn.SimpleMLP(input_size, hidden_sizes, output_size,
                                    active_end=active_end,dropout=dropout)

    def forward(self, tx:torch.Tensor,obj:torch.Tensor,zpar:torch.Tensor,zimg:torch.Tensor,) -> Tuple[torch.Tensor,None]:
        """
        Forward pass of the model.

        Args:
            tx:     Time+State tensor.
            obj:    Objective tensor.
            zpar:   Parameter tensor.
            zimg:   Image tensor.

        Returns:
            ycm:  Output tensor.
            zcm:  None.
        """

        # Command MLP
        xcm = torch.cat((tx,obj,zpar,zimg),-1)
        ycm = self.network(xcm)

        return ycm,None

class VisionMLP(nn.Module):
    def __init__(self,
                 state:List[int],
                 state_layer:int,
                 hidden_sizes:List[int],
                 output_size:int,
                 visual_type:str='SqueezeNet1_1'):
        """
        Initialize a vision MLP model.

        Args:
            state:          List of state indices.
            state_layer:    Layer where the state is inserted.
            hidden_sizes:   List of hidden layer sizes.
            output_size:    Output size.
            visual_type:    Type of visual
            
        Variables:
            networks:       List of neural networks.
        """

        # Initialize the parent class
        super(VisionMLP, self).__init__()

        # Some useful intermediate variables
        state_size = len(state)
        visual_network = bn.VisionCNN(visual_type)

        if state_layer == 0:
            mlp_networks = [nn.Identity(),
                            bn.SimpleMLP(visual_network.Nout+state_size,hidden_sizes,output_size) ]
        else:
            mlp_networks = [bn.SimpleMLP(visual_network.Nout,hidden_sizes[:state_layer-1],hidden_sizes[state_layer-1]),
                            bn.SimpleMLP(hidden_sizes[state_layer-1]+state_size,hidden_sizes[state_layer:],output_size) ]
        
        # Define the model
        self.ix_tx = np.array(state)
        self.networks = nn.ModuleList(
            [visual_network] + 
            mlp_networks)
    
    def forward(self, img:torch.Tensor,tx:torch.Tensor) ->  Tuple[torch.Tensor,None]:
        """
        Forward pass of the model.

        Args:
            img:  Image tensor.
            tx:   Time+State tensor.

        Returns:
            yvs:  Output tensor.
            zvs:  None.
        """

        # Check if the image is a batch
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Vision CNN
        yim = self.networks[0](img).squeeze()

        # Vision MLPs
        xvs = self.networks[1](yim)
        yvs = self.networks[2](torch.cat((xvs,tx),-1))

        return yvs,None

    def set_vision_prompt(self, text: str):
        # network[0] is the vision backbone (bn.VisionCNN instance or our CLIPSeg encoder inside it)
        vis = self.networks[0]
        # If it's the CLIPSegSoftmask wrapper, drill into .networks (the actual encoder) and set prompt
        if hasattr(vis, "networks") and hasattr(vis.networks, "set_prompt"):
            vis.networks.set_prompt(text)

class VisionEncoder(nn.Module):
    def __init__(self,
                 state:List[int],
                 state_layer:int,
                 hidden_sizes:List[int],
                 encoder_size:int, decoder_size:int,
                 visual_type:str='SqueezeNet1_1'):
        """
        Initialize a vision MLP model.

        Args:
            state:          List of state indices.
            state_layer:    Layer where the state is inserted.
            hidden_sizes:   List of hidden layer sizes.
            encoder_size: Encoder output size.
            decoder_size: Decoder output size.
            visual_type:    Type of visual network.

        Variables:
            networks:       List of neural networks.
        """

        # Initialize the parent class
        super(VisionEncoder, self).__init__()

        # Some useful intermediate variables
        state_size = len(state)
        visual_network = bn.VisionCNN(visual_type)

        if state_layer == 0:
            mlp_networks = [nn.Identity(),
                            bn.SharpEncoder(visual_network.Nout+state_size,hidden_sizes,encoder_size,decoder_size)]
        else:
            mlp_networks = [bn.SimpleMLP(visual_network.Nout,hidden_sizes[:state_layer-1],hidden_sizes[state_layer-1]),
                            bn.SharpEncoder(hidden_sizes[state_layer-1]+state_size,hidden_sizes[state_layer:],encoder_size,decoder_size) ]

        # Define the model
        self.ix_tx = np.array(state)
        self.networks = nn.ModuleList(
            [visual_network] + 
            mlp_networks)
        
    def forward(self, img:torch.Tensor,tx:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            img:  Image tensor.
            tx:   Time+State tensor.

        Returns:
            yvs:  Decoder output tensor.
            zvs:  Encoder output tensor.
        """

        
        # Check if the image is a batch
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Vision CNN
        yim = self.networks[0](img).squeeze()

        # Vision MLPs
        xvs = self.networks[1](yim)
        yvs,zvs = self.networks[2](torch.cat((xvs,tx),-1))

        return yvs,zvs