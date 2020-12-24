from PIL import Image
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F

from models.nn.resnet import Resnet

# From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Number of Linear input connections depends on output of conv2d layers
# and therefore the input image size, so compute it.
def conv2d_size_out(size, kernel_size=8, stride=4):
    return (size - (kernel_size - 1) - 1) // stride  + 1

class LateFusion(nn.Module):
    def __init__(self, visual_model, object_embeddings, policy_model):
        super(LateFusion, self).__init__()
        self.visual_model = visual_model
        self.object_embeddings = object_embeddings
        self.policy_model = policy_model

    def forward(self, frames, object_index):
        if isinstance(self.visual_model, Resnet):
            unstacked_visual_outputs = []
            for unstacked_frames in torch.split(frames,
                    split_size_or_sections=3, dim=1):
                # Need to turn tensors into PIL images
                # Cast to uint8 first to reduce amount of memory copied, and
                # transpose to put RGB channels in the last dimension
                # Channels should be in RGB order
                unstacked_frames = [Image.fromarray(frame.to(dtype=torch.uint8)
                    .cpu().numpy().transpose(1, 2, 0)) for frame in
                    unstacked_frames]
                unstacked_visual_outputs.append(self.visual_model.featurize(
                    unstacked_frames))
            # unstacked_visual_outputs is now length frame_stack, each element
            # is shape (batch_size, 3, 300, 300)
            visual_output = torch.cat(unstacked_visual_outputs, dim=1)
        else:
            visual_output = self.visual_model(frames)
        embedded_object = self.object_embeddings(object_index)
        return self.policy_model(visual_output, embedded_object)

class NatureCNN(nn.Module):
    def __init__(self, frame_stack=1, frame_width=300, frame_height=300):
        super(NatureCNN, self).__init__()
        self.frame_stack = frame_stack

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(frame_width,
            kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3,
            stride=1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(frame_height,
            kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3,
            stride=1)
        self.output_size = convw * convh * 64

        # Nature architecture with batchnorm
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3*self.frame_stack,
            out_channels=32, kernel_size=8, stride=4, bias=True),
            nn.BatchNorm2d(32), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64,
            kernel_size=4, stride=2, bias=True),
            nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64,
            kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(64), nn.ReLU())

    def forward(self, frames):
        x = self.conv1(frames)
        x1 = self.conv2(x)
        x2 = self.conv3(x1)

        return x2

class FCPolicy(nn.Module):
    def __init__(self, input_size, num_actions, num_fc_layers=1):
        super(FCPolicy, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.num_fc_layers = num_fc_layers
        self.fc_units = [512, 256, 128, 64]
        self.fc_layers = nn.ModuleList()
        for i in range(num_fc_layers):
            if i == 0:
                in_features = self.input_size
            else:
                in_features = self.fc_units[i-1]
            self.fc_layers.append(nn.Sequential(
                nn.Linear(in_features=in_features,
                    out_features=self.fc_units[i], bias=True), nn.ReLU()))
        self.action_logits = nn.Sequential(
                nn.Linear(in_features=self.fc_units[i],
                    out_features=self.num_actions, bias=True))

    def forward(self, visual_output, object_embedding):
        # "Late Fusion"
        # Reshape conv output to (N, -1) and concatenate object embedding
        # This reshaping will also work for non-conv features
        x  = torch.cat([visual_output.view(visual_output.size(0), -1),
            object_embedding], -1)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        action_probs = self.action_logits(x)

        return action_probs

class ObjectEmbedding(nn.Module):
    def __init__(self, num_objects, object_embedding_dim):
        super(ObjectEmbedding, self).__init__()
        self.num_objects = num_objects
        self.object_embedding_dim = object_embedding_dim
        self.object_embedding = nn.Embedding(num_embeddings=self.num_objects,
                embedding_dim=self.object_embedding_dim)

    def forward(self, object_index):
        return self.object_embedding(object_index)

