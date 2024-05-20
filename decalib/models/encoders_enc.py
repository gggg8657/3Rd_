# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import resnet

class ResnetEncoder(nn.Module):
    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__()
        feature_size = 2048
        self.encoder = resnet.load_ResNet50Model() #out: 2048
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )
        self.last_op = last_op

    def forward(self, inputs):
        features = self.encoder(inputs)
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters

class ResnetEncoder_feat(nn.Module):
    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder_feat, self).__init__()
        feature_size = 2048
        self.encoder = resnet.load_ResNet50Model() #out: 2048
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )
        self.last_op = last_op

    def forward(self, inputs):
        features = self.encoder(inputs)
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return features

# from transformers import BertModel, BertConfig

class DetailAUTransformer(nn.Module):
    def __init__(self, detail_dim=2048, au_dim=41, hidden_dim=256, output_dim=128, num_layers=4, num_heads=8):
        super(DetailAUTransformer, self).__init__()
        
        # Embedding layers
        self.detail_embedding = nn.Linear(detail_dim, hidden_dim)
        self.au_embedding = nn.Linear(au_dim, hidden_dim)
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder for detail parameters
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, detail_features, au_features):
        batch_size = detail_features.size(0)
        
        # Embedding
        detail_embedded = self.detail_embedding(detail_features)
        au_embedded = self.au_embedding(au_features)
        
        # Combine and add positional embeddings
        x = torch.stack([detail_embedded, au_embedded], dim=1)
        x += self.pos_embedding
        
        # Transformer Encoder
        x = x.permute(1, 0, 2)  # (sequence_length, batch_size, hidden_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, sequence_length, hidden_dim)
        
        # Use only the [CLS] token (first token) for regression
        x = x[:, 0, :]
        
        # Decode to detail parameters
        detail_params = self.decoder(x)
        
        return detail_params

class AUEncoder(nn.Module):
    def __init__(self):
        super(AUEncoder,self).__init__()

        self.flatten=nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(27*512, 50)

    def forward(self, x):
        x=self.flatten(x)
        x=self.fc(x)
        return x