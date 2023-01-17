from __future__ import division, print_function, absolute_import

import pdb
import copy
from collections import OrderedDict
from torch.nn import functional as F
import torch
import torch.nn as nn
import numpy as np
import math
    
class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val
        self.fc1 = nn.Linear(dim_input, dim_val, bias = True)
    
    def forward(self, x):
        x = self.fc1(x)
        
        return x

class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = True)
    
    def forward(self, x):
        x = self.fc1(x)
        
        return x

class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn
        
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = True)
    
    def forward(self, x):
        x = self.fc1(x)
        
        return x

class Learner(nn.Module):

    def __init__(self, image_size, bn_eps, bn_momentum, n_classes):
        super(Learner, self).__init__()
        self.model = nn.ModuleDict({'features1': nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, padding=1)),
            ('norm1', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
            ('relu1', nn.ReLU(inplace=False)),
            ('pool1', nn.MaxPool2d(2)),
            
            ('conv2', nn.Conv2d(32, 32, 3, padding=1)),
            ('norm2', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
            ('relu2', nn.ReLU(inplace=False)),
            ('pool2', nn.MaxPool2d(2)),
            
            ('conv3', nn.Conv2d(32, 32, 3, padding=1)),
            ('norm3', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
            ('relu3', nn.ReLU(inplace=False)),
            ('pool3', nn.MaxPool2d(2)),
            ]))
        ,'features2': nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(32, 32, 3, padding=1)),
            ('norm4', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
            ('relu4', nn.ReLU(inplace=False)),
            ]))
        ,'features3': nn.Sequential(OrderedDict([
            ('pool3', nn.MaxPool2d(2)),
            ]))                 
        })

        clr_in = image_size // 2**4
        
        self.model.update({'cls': nn.Linear(32*clr_in*clr_in, n_classes)})

        self.criterion = nn.CrossEntropyLoss()
        
        self.model.update({'value': nn.Linear(252, 252)})
        self.model.update({'key': nn.Linear(252, 252)})
        self.model.update({'query': nn.Linear(252, 252)})
        
        self.model.update({'dense': nn.Linear(252, 252)})
        self.model.update({'layernorm': nn.LayerNorm(252, bn_eps)})
    
    def forward(self, x, x_t):
        batch_size, c, h, w=x.size()
        batch_size_t, _, _, _=x_t.size()

        x = x.reshape([batch_size,h,w*c])

        if len(x_t)==3:
            x_t = x_t.reshape([h,w*c])
        else:
            x_t = x_t.reshape([batch_size_t,h,w*c])

        v_h = self.model.value(x)
        k_h = self.model.key(x)
        q_h = self.model.query(x_t)

        dots = torch.matmul(q_h, k_h.transpose(-1, -2))
        dots = dots / math.sqrt(w)
        dots = nn.Softmax(dim=-1)(dots)

        out = torch.matmul(dots, v_h)
        hidden_states = self.model.dense(out)
        hidden_states = self.model.layernorm(hidden_states+x)

        x = x_t + hidden_states
        x = x.reshape([batch_size, c, h, w])
        
        x_f = self.model.features1(x)
        x_f_2 = self.model.features2(x_f)
        x_f_3 = self.model.features3(x_f_2)

        x = torch.reshape(x_f_3, [x_f_3.size(0), -1])
        outputs = self.model.cls(x)

        return outputs , x_f_2

    def get_flat_params(self):
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen

    def reset_batch_stats(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                
                
