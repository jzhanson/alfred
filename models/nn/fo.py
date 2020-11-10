import torch
from torch import nn, optim
from torch.nn import functional as F

# From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Number of Linear input connections depends on output of conv2d layers
# and therefore the input image size, so compute it.
def conv2d_size_out(size, kernel_size=8, stride=4):
    return (size - (kernel_size - 1) - 1) // stride  + 1

class NatureCNN(nn.Module):
    def __init__(self, num_objects, num_actions, frame_stack=3, object_embedding_dim=16,
            frame_width=300, frame_height=300):
        super(NatureCNN, self).__init__()
        self.num_objects = num_objects
        self.num_actions = num_actions
        self.frame_stack = frame_stack
        self.object_embedding_dim = object_embedding_dim

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(frame_width,
            kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3,
            stride=1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(frame_height,
            kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3,
            stride=1)
        linear_input_size = convw * convh * 64 + self.object_embedding_dim

        self.object_embedding = nn.Embedding(num_embeddings=self.num_objects,
                embedding_dim=self.object_embedding_dim)

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
        self.fc1 = nn.Sequential(nn.Linear(in_features=linear_input_size,
            out_features=512, bias=True), nn.ReLU())
        self.action_logits = nn.Sequential(nn.Linear(in_features=512,
            out_features=self.num_actions, bias=True))

        # Layers declaration - model inspired by "Learning About Objects by
        # Learning to Interact with Them", https://arxiv.org/abs/2006.09306
        ''' self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,
        out_channels=32,
            kernel_size=5, stride=3, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, padding=2,
            stride=2, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, padding=2,
            stride=2, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=5,
            padding=2, stride=2, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        '''

    def forward(self, frames, obj_indexes):
        x = self.conv1(frames)
        x1 = self.conv2(x)
        x2 = self.conv3(x1)

        # "Late Fusion"
        object_embed = self.object_embedding(obj_indexes)
        # Reshape conv output to (N, -1) and concatenate object one hots
        x3 = self.fc1(torch.cat([x2.view(x2.size(0), -1), object_embed], -1))
        action_probs = self.action_logits(x3)

        return action_probs

