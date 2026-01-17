import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    alexnet,
    squeezenet1_1,
    resnet18,
    vgg11,
    vit_b_16,
    AlexNet_Weights,
    SqueezeNet1_1_Weights,
    ResNet18_Weights,
    VGG11_Weights,
    ViT_B_16_Weights
)
from typing import List,Tuple
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


class SimpleMLP(nn.Module):
    def __init__(self, input_size:int, hidden_sizes:List[int],
                 output_size:int,
                 active_end=False, dropout=0.2):
        """
        Initialize a simple MLP model.

        Args:
            input_size:     Input size.
            hidden_sizes:   List of hidden layer sizes.
            output_size:    Output size.
            active_end:     Activation function at the end.
            dropout:        Dropout rate.

        Variables:
            networks:       List of neural networks.
        """
        # Initialize the parent class
        super(SimpleMLP, self).__init__()

        # Populate the layers
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))

        # Add final activation function if required
        if active_end:
            layers.append(nn.ReLU())

        # Define the model
        self.networks = nn.Sequential(*layers)
    
    def forward(self, xnn:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn:  Input tensor.

        Returns:
            ynn:  Output tensor.
        """

        # Simple MLP
        ynn = self.networks(xnn)         

        return ynn
    
class SimpleEncoder(nn.Module):
    def __init__(self, input_size:int, hidden_sizes:List[int],
                 encoder_output_size:int, decoder_output_size:int,
                 active_end=False, dropout=0.2):
        """
        Initialize a simple encoder model (with a decoder for training).

        Args:
            input_size:         Input size.
            hidden_sizes:       List of hidden layer sizes.
            encoder_output_size: Encoder output size.
            decoder_output_size: Decoder output size.
            active_end:         Activation function at the end.
            dropout:            Dropout rate.

        Variables:
            networks:           List of neural networks.
            final_layer:        Final layer.
        """
        # Initialize the parent class
        super(SimpleEncoder, self).__init__()

        # Populate the layers
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
        layers.append(nn.Linear(prev_size, encoder_output_size))
        layers.append(nn.ReLU())

        # Add final activation function if required
        if active_end:
            layers.append(nn.ReLU())

        # Define the model
        self.networks = nn.Sequential(*layers)
        self.final_layer = nn.Linear(encoder_output_size, decoder_output_size)

    def forward(self, xnn:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            xnn:  Input tensor.

        Returns:
            ynn:  Decoder output tensor.
            znn:  Encoder output tensor.
        """

        # Simple Encoder
        znn = self.networks(xnn)            # encoder output
        ynn = self.final_layer(znn)         # decoder output
        
        return ynn,znn

class SharpEncoder(nn.Module):
    def __init__(self, input_size:int, hidden_sizes:List[int],
                 encoder_output_size:int, decoder_output_size:int,
                 active_end=False, base_dropout=0.1, final_dropout=0.4):
        """
        Initialize a simple encoder model (with a decoder for training).

        Args:
            input_size:         Input size.
            hidden_sizes:       List of hidden layer sizes.
            encoder_output_size: Encoder output size.
            decoder_output_size: Decoder output size.
            active_end:         Activation function at the end.
            dropout:            Dropout rate.

        Variables:
            networks:           List of neural networks.
            final_layer:        Final layer.
        """
        # Initialize the parent class
        super(SharpEncoder, self).__init__()

        # Populate the layers
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(base_dropout))
            prev_size = size
        layers.append(nn.Linear(prev_size, encoder_output_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(final_dropout))

        # Add final activation function if required
        if active_end:
            layers.append(nn.ReLU())

        # Define the model
        self.networks = nn.Sequential(*layers)
        self.final_layer = nn.Linear(encoder_output_size, decoder_output_size)

    def forward(self, xnn:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            xnn:  Input tensor.

        Returns:
            ynn:  Decoder output tensor.
            znn:  Encoder output tensor.
        """

        # Simple Encoder
        znn = self.networks(xnn)            # encoder output
        ynn = self.final_layer(znn)         # decoder output
        
        return ynn,znn

class DirectEncoder(nn.Module):
    def __init__(self, input_size:int, hidden_sizes:List[int],
                 encoder_output_size:int,
                 active_end=False, dropout=0.2):
        """
        Initialize a simple encoder model (with a decoder for training).

        Args:
            input_size:         Input size.
            hidden_sizes:       List of hidden layer sizes.
            decoder_output_size: Decoder output size.
            active_end:         Activation function at the end.
            dropout:            Dropout rate.

        Variables:
            networks:           List of neural networks.
            final_layer:        Final layer.
        """
        # Initialize the parent class
        super(DirectEncoder, self).__init__()

        # Populate the layers
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())

            if size != hidden_sizes[-1]:
                layers.append(nn.Dropout(dropout))
                
            prev_size = size

        layers.append(nn.Linear(prev_size, encoder_output_size))

        # Add final activation function if required
        if active_end:
            layers.append(nn.ReLU())

        # Define the model
        self.networks = nn.Sequential(*layers)

    def forward(self, xnn:torch.Tensor) -> Tuple[torch.Tensor,None]:
        """
        Forward pass of the model.

        Args:
            xnn:  Input tensor.

        Returns:
            ynn:  Decoder output tensor.
            znn:  Encoder output tensor.
        """

        # Simple Encoder
        ynn = self.networks(xnn)            # encoder output
        
        return ynn,None

class WideShallowCNN(nn.Module):
    def __init__(self, input_channels=3, output_dim=1000):
        super(WideShallowCNN, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)  # 224x224x3 -> 224x224x64
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # 112x112x64 -> 112x112x128
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # 56x56x128 -> 56x56x256

        # Fully Connected Layer
        # Flatten the features before passing to this layer
        # Assuming input image size is 224x224, this will need to be adjusted if input size changes
        self.fc1 = nn.Linear(256 * 28 * 28, 1000)

        # Max Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        # Apply Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))  # 224x224x64 -> 112x112x64
        # Apply Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # 112x112x128 -> 56x56x128
        # Apply Conv3 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x)))  # 56x56x256 -> 28x28x256

        # Flatten the feature map
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256*28*28)

        # Fully Connected Layer
        x = self.fc1(x)  # Output 1000 features

        return x
    
class VisionCNN(nn.Module):
    def __init__(self, visual_type:str,
                 Nout:int=1000):
        """
        Initialize a vision CNN model.

        Args:
            visual_type:    Type of visual network.
            Nout:           Output size.

        Variables:
            networks:       Vision network.
            Nout:           Output size.
        """
        # Initialize the parent class
        super(VisionCNN, self).__init__()

        # Instantiate Visual Network
        if visual_type == "AlexNet":
            networks = alexnet(weights=AlexNet_Weights.DEFAULT)
        elif visual_type == "SqueezeNet1_1":
            networks = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
        elif visual_type == "ResNet18":
            networks = resnet18(weights=ResNet18_Weights.DEFAULT)()
        elif visual_type == "VGG11":
            networks = vgg11(weights=VGG11_Weights.DEFAULT)
        elif visual_type == "ViT_B_16":
            networks = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        elif visual_type == "WideShallowCNN":
            networks = WideShallowCNN()
        elif visual_type == "CLIPSegSoftmask":
            # Default to 256-D output unless overridden by Nout arg
            networks = CLIPSegSoftmaskEncoder(Nout=Nout)
        else:
            raise ValueError(f"Invalid visual_type: {visual_type}")
        
        # Define the model
        self.networks = networks
        self.Nout = Nout

    def forward(self, xnn:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            xnn:  Input tensor.
        
        Returns:
            ynn:  Output tensor.
        """

        # Vision CNN
        ynn = self.networks(xnn)

        return ynn
    
class CLIPSegSoftmaskEncoder(nn.Module):
    """
    Runs CLIPSeg and turns (logits + ViT patch tokens) into a single Nout-dim latent.
    - Prompt-aware via decoder logits, but outputs ONE vector (not a mask).
    """
    def __init__(self, hf_model="CIDAS/clipseg-rd64-refined", Nout=256, freeze=True, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPSegProcessor.from_pretrained(hf_model)
        self.model = CLIPSegForImageSegmentation.from_pretrained(hf_model).to(self.device).eval()
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        Dv = self.model.config.vision_config.hidden_size  # e.g., 768
        self.proj = nn.Linear(Dv, Nout)
        self.Nout = Nout
        self._prompt_ids = None
        self._attn_mask = None

    def set_prompt(self, text: str):
        # Pre-tokenize once; avoid per-frame re-tokenization overhead.
        toks = self.processor.tokenizer(
            [text], padding=True, return_tensors="pt"
        )
        self._prompt_ids  = toks["input_ids"].to(self.device)
        self._attn_mask   = toks["attention_mask"].to(self.device)

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B,3,H,W) float tensor in [0,1] or uint8; returns (B, Nout).
        """
        assert self._prompt_ids is not None, "Call set_prompt(text) once before forward()."

        # Let the HF processor build pixel_values (handles resize/normalize expected by CLIPSeg)
        # Convert batch tensor → list of PIL or np for the processor; do per-batch for simplicity.
        B = img.shape[0]
        imgs = []
        if img.dtype != torch.uint8:
            x = (img.clamp(0,1)*255).byte().detach().cpu()
        else:
            x = img.detach().cpu()
        for b in range(B):
            imgs.append(x[b].permute(1,2,0).numpy())  # HWC uint8

        proc = self.processor(images=imgs, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(self.device)

        # Forward with full outputs to access both logits and vision tokens
        out = self.model(
            input_ids=self._prompt_ids.expand(B, -1),
            attention_mask=self._attn_mask.expand(B, -1),
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True
        )
        # vision tokens: [B, 1+N, Dv] → drop CLS
        v_tokens = out.vision_model_output.last_hidden_state[:, 1:, :]  # [B, N, Dv]
        B, N, Dv = v_tokens.shape
        side = int(N ** 0.5)

        # decoder logits → soft mask in [0,1], then pool to patch grid
        logits = out.logits  # [B, H, W] or [B,1,H,W]
        if logits.ndim == 4:
            logits = logits[:,0]
        soft = torch.sigmoid(logits).unsqueeze(1)                 # [B,1,H,W]
        weights = F.adaptive_avg_pool2d(soft, (side, side))       # [B,1,side,side]
        weights = weights.flatten(1).unsqueeze(-1)                # [B,N,1]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # weighted average of tokens -> [B, Dv], then project to Nout
        z = (weights * v_tokens).sum(dim=1)                       # [B,Dv]
        return self.proj(z)                                       # [B,Nout]