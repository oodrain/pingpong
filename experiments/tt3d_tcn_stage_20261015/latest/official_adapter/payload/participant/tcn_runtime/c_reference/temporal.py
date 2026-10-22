"""Unchanged temporal block used by the frozen model."""
import torch
from torch import nn

class TemporalBlock(nn.Module):
    def __init__(self,channels,dilation,dropout):
        super().__init__();self.conv=nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation)
        self.norm=nn.LayerNorm(channels);self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        y=self.conv(x);y=self.norm(y.transpose(1,2)).transpose(1,2)
        return torch.relu(x+self.dropout(torch.relu(y)))
