import torch
import torch.nn as nn
from funasr.models.transformer.utils.nets_utils import get_activation


class HubertNoLayerNormConvLayer(nn.Module):
    def __init__(self,
                 layer_id=0,
                 conv_dim=[512, 512, 512, 512, 512, 512, 512],
                 conv_kernel=[10, 3, 3, 3, 3, 2, 2],
                 conv_stride=[ 5, 2, 2, 2, 2, 2, 2],
                 conv_bias=False,
                 conv_activation = None
                 ):
        super().__init__()
        self.in_conv_dim = conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=conv_kernel[layer_id],
            stride=conv_stride[layer_id],
            bias=conv_bias,
        )
        if conv_activation is not None:
            self.activation = get_activation(conv_activation)
        else:
            self.activation = None

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        return hidden_states


class HubertGroupNormConvLayer(nn.Module):
    def __init__(self,
                 layer_id=0,
                 conv_dim=[512, 512, 512, 512, 512, 512, 512],
                 conv_kernel=[10, 3, 3, 3, 3, 2, 2],
                 conv_stride=[5, 2, 2, 2, 2, 2, 2],
                 conv_bias=False,
                 conv_activation=None
                 ):
        super().__init__()
        self.in_conv_dim = conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=conv_kernel[layer_id],
            stride=conv_stride[layer_id],
            bias=conv_bias,
        )
        if conv_activation is not None:
            self.activation = get_activation(conv_activation)
        else:
            self.activation = None

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        return hidden_states


class HubertLayerNormConvLayer(nn.Module):
    def __init__(self,
                 layer_id=0,
                 conv_dim=[512, 512, 512, 512, 512, 512, 512],
                 conv_kernel=[10, 3, 3, 3, 3, 2, 2],
                 conv_stride=[ 5, 2, 2, 2, 2, 2, 2],
                 conv_bias=False,
                 conv_activation = None
                 ):
        super().__init__()
        self.in_conv_dim = conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=conv_kernel[layer_id],
            stride=conv_stride[layer_id],
            bias=conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        if conv_activation is not None:
            self.activation = get_activation(conv_activation)
        else:
            self.activation = None

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        return hidden_states


class HubertFeatureProjection(nn.Module):
    def __init__(
            self,
            last_conv_dim,
            output_dim,
            feat_proj_layer_norm = True,
            feat_proj_dropout = 0.0,
            feat_extract_norm="group",
            num_feat_extract_layers= 7,
            ):
        super().__init__()
        layer_norm_eps = 1e-5
        self.feat_proj_layer_norm = feat_proj_layer_norm
        if self.feat_proj_layer_norm:
            self.layer_norm = nn.LayerNorm(last_conv_dim, eps=layer_norm_eps)
        self.projection = nn.Linear(last_conv_dim, output_dim)
        self.dropout = nn.Dropout(feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states