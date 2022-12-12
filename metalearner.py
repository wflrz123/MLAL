from __future__ import division, print_function, absolute_import

import pdb
import math
import torch
import torch.nn as nn


class MetaLSTMCell(nn.Module):
    """C_t = f_t * C_{t-1} + i_t * \tilde{C_t}"""
    def __init__(self, input_size, hidden_size, n_learner_params):
        super(MetaLSTMCell, self).__init__()
        """Args:
            input_size (int): cell input size, default = 20
            hidden_size (int): should be 1
            n_learner_params (int): number of learner's parameters
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_learner_params = n_learner_params
        self.WF = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.WI = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.cI = nn.Parameter(torch.Tensor(n_learner_params, 1))
        self.bI = nn.Parameter(torch.Tensor(1, hidden_size))
        self.bF = nn.Parameter(torch.Tensor(1, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.01, 0.01)

        # want initial forget value to be high and input value to be low so that 
        #  model starts with gradient descent
        nn.init.uniform_(self.bF, 4, 6)
        nn.init.uniform_(self.bI, -5, -4)

    def init_cI(self, flat_params):
        self.cI.data.copy_(flat_params.unsqueeze(1))

    def forward(self, inputs, hx=None):
        """Args:
            inputs = [x_all, grad]:
                x_all (torch.Tensor of size [n_learner_params, input_size]): outputs from previous LSTM
                grad (torch.Tensor of size [n_learner_params]): gradients from learner
            hx = [f_prev, i_prev, c_prev]:
                f (torch.Tensor of size [n_learner_params, 1]): forget gate
                i (torch.Tensor of size [n_learner_params, 1]): input gate
                c (torch.Tensor of size [n_learner_params, 1]): flattened learner parameters
        """
        x_all, grad = inputs
#         print(x_all.shape,grad.shape)
        batch, _ = x_all.size()

        if hx is None:
            f_prev = torch.zeros((batch, self.hidden_size)).to(self.WF.device)
            i_prev = torch.zeros((batch, self.hidden_size)).to(self.WI.device)
            c_prev = self.cI
            hx = [f_prev, i_prev, c_prev]

        f_prev, i_prev, c_prev = hx

        # f_t = sigmoid(W_f * [grad_t, loss_t, theta_{t-1}, f_{t-1}] + b_f)
        f_next = torch.mm(torch.cat((x_all, c_prev, f_prev), 1), self.WF) + self.bF.expand_as(f_prev)
        # i_t = sigmoid(W_i * [grad_t, loss_t, theta_{t-1}, i_{t-1}] + b_i)
        i_next = torch.mm(torch.cat((x_all, c_prev, i_prev), 1), self.WI) + self.bI.expand_as(i_prev)
        # next cell/params
        c_next = torch.sigmoid(f_next).mul(c_prev) - torch.sigmoid(i_next).mul(grad)
        
        return c_next, [f_next, i_next, c_next]

    def extra_repr(self):
        s = '{input_size}, {hidden_size}, {n_learner_params}'
        return s.format(**self.__dict__)

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
    
# class MetaLearner(nn.Module):

#     def __init__(self, input_size, hidden_size, n_learner_params):
#         super(MetaLearner, self).__init__()
#         """Args:
#             input_size (int): for the first LSTM layer, default = 4
#             hidden_size (int): for the first LSTM layer, default = 20
#             n_learner_params (int): number of learner's parameters
#         """
#         self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
#         self.metalstm = MetaLSTMCell(input_size=input_size, hidden_size=1, n_learner_params=n_learner_params)
# #         self.value = Value(hidden_size, hidden_size)
# #         self.key = Key(hidden_size, 4)
# #         self.query = Query(hidden_size, 4)
# #         self.attend = nn.Softmax(dim = -1)

#     def forward(self, inputs, hs=None):
#         """Args:
#             inputs = [loss, grad_prep, grad]
#                 loss (torch.Tensor of size [1, 2])
#                 grad_prep (torch.Tensor of size [n_learner_params, 2])
#                 grad (torch.Tensor of size [n_learner_params])

#             hs = [(lstm_hn, lstm_cn), [metalstm_fn, metalstm_in, metalstm_cn]]
#         """

#         loss, grad_prep, grad = inputs
# #         print(loss)
#         loss = loss.expand_as(grad_prep)
#         inputs = torch.cat((loss, grad_prep), 1)   # [n_learner_params, 4]

#         if hs is None:
#             hs = [None, None]

#         lstmhx, lstmcx = self.lstm(inputs, hs[0])
        
# #         v_h = self.value(inputs)
# #         k_h = self.key(inputs)
# #         q_h = self.query(inputs)
# # #         print(q_h.shape, k_h.shape)
# #         dots = torch.matmul(q_h, k_h.transpose(-1, -2)) * 64 ** -0.5
        
# #         attn = self.attend(dots)
# # #         print(attn.shape, v_h.shape)
# #         out = torch.matmul(attn, v_h)

# # #         print(lstmhx.shape, out.shape)
# #         attention_h = inputs + out
# #         print(lstmcx.shape)
# #         print(len(lstmhx))
# #         flat_learner_unsqzd, metalstm_hs = self.metalstm([inputs, grad], hs[1])
# # #         print(len(metalstm_hs[0]))
# #         return flat_learner_unsqzd.squeeze(), [(lstmhx, lstmcx), metalstm_hs]

#         flat_learner_unsqzd, metalstm_hs = self.metalstm([lstmhx, grad], hs[1])
# #         print(len(metalstm_hs[0]))
#         return flat_learner_unsqzd.squeeze(), [(lstmhx, lstmcx), metalstm_hs]

# class MetaLearner(nn.Module):

#     def __init__(self, input_size, hidden_size, n_learner_params):
#         super(MetaLearner, self).__init__()
#         """Args:
#             input_size (int): for the first LSTM layer, default = 4
#             hidden_size (int): for the first LSTM layer, default = 20
#             n_learner_params (int): number of learner's parameters
#         """
#         self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
#         self.metalstm = MetaLSTMCell(input_size=hidden_size, hidden_size=1, n_learner_params=n_learner_params)
#         self.value = Value(571, 571)
#         self.key = Key(571, 20)
#         self.query = Query(571, 20)
#         self.attend = nn.Softmax(dim = -1)
#         self.linear = nn.Linear(in_features=4, out_features=20)

#     def forward(self, inputs, hs=None):
#         """Args:
#             inputs = [loss, grad_prep, grad]
#                 loss (torch.Tensor of size [1, 2])
#                 grad_prep (torch.Tensor of size [n_learner_params, 2])
#                 grad (torch.Tensor of size [n_learner_params])

#             hs = [(lstm_hn, lstm_cn), [metalstm_fn, metalstm_in, metalstm_cn]]
#         """

#         loss, grad_prep, grad = inputs
# #         print(loss)
#         loss = loss.expand_as(grad_prep)
#         inputs = torch.cat((loss, grad_prep), 1)   # [n_learner_params, 4

#         if hs is None:
#             hs = [None, None]

#         lstmhx, lstmcx = self.lstm(inputs, hs[0])
# #         print(inputs.shape)
#         hx = torch.reshape(lstmhx, (748,571))
# #         print(hx[0][0],hx[1][0],hx[2][0])
#         v_h = self.value(hx)
#         k_h = self.key(hx)
#         q_h = self.query(hx)
#         dots = torch.matmul(q_h, k_h.transpose(-1, -2)) * 571 ** -0.5
# #         print(dots[0][0],dots[1][0],dots[2][0])
# #         attn = self.attend(dots)
# #         print(attn)
#         out = torch.matmul(dots, v_h)
# #         print(hx[0][0],out[0][0])
#         attention_h = hx + out
# #         print("att:",attention_h.shape)
# #         att = torch.flatten(attention_h, 0).unsqueeze(1)
#         att = torch.reshape(attention_h, (106777,4))

# #         print(att.shape)
#         flat_learner_unsqzd, metalstm_hs = self.metalstm([att, grad], hs[1])
# #         print(len(metalstm_hs[0]))
#         return flat_learner_unsqzd.squeeze(), [(lstmhx, lstmcx), metalstm_hs]

# class MetaLearner(nn.Module):

#     def __init__(self, input_size, hidden_size, n_learner_params):
#         super(MetaLearner, self).__init__()
#         """Args:
#             input_size (int): for the first LSTM layer, default = 4
#             hidden_size (int): for the first LSTM layer, default = 20
#             n_learner_params (int): number of learner's parameters
#         """
#         self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
#         self.metalstm = MetaLSTMCell(input_size=hidden_size, hidden_size=1, n_learner_params=n_learner_params)
#         self.value = Value(132, 132)
#         self.key = Key(132, 20)
#         self.query = Query(132, 20)
#         self.attend = nn.Softmax(dim = -1)
#         self.linear = nn.Linear(in_features=4, out_features=20)

#     def forward(self, inputs, hs=None):
#         """Args:
#             inputs = [loss, grad_prep, grad]
#                 loss (torch.Tensor of size [1, 2])
#                 grad_prep (torch.Tensor of size [n_learner_params, 2])
#                 grad (torch.Tensor of size [n_learner_params])

#             hs = [(lstm_hn, lstm_cn), [metalstm_fn, metalstm_in, metalstm_cn]]
#         """

#         loss, grad_prep, grad = inputs
# #         print(loss)
#         loss = loss.expand_as(grad_prep)
#         inputs = torch.cat((loss, grad_prep), 1)   # [n_learner_params, 4

#         if hs is None:
#             hs = [None, None]

#         lstmhx, lstmcx = self.lstm(inputs, hs[0])
# #         print(inputs.shape)
# #         hx = torch.reshape(lstmhx, (997,132))
# # #         print(hx[0][0],hx[1][0],hx[2][0])
# #         v_h = self.value(hx)
# #         k_h = self.key(hx)
# #         q_h = self.query(hx)
# #         dots = torch.matmul(q_h, k_h.transpose(-1, -2)) * 132 ** -0.5
# # #         print(dots[0][0],dots[1][0],dots[2][0])
# # #         attn = self.attend(dots)
# # #         print(attn)
# #         out = torch.matmul(dots, v_h)
# # #         print(hx[0][0],out[0][0])
# #         attention_h = hx + out
# # #         print("att:",attention_h.shape)
# # #         att = torch.flatten(attention_h, 0).unsqueeze(1)
# #         att = torch.reshape(attention_h, (32901,4))

# #         print(att.shape)
#         flat_learner_unsqzd, metalstm_hs = self.metalstm([lstmhx, grad], hs[1])
# #         print(len(metalstm_hs[0]))
#         return flat_learner_unsqzd.squeeze(), [(lstmhx, lstmcx), metalstm_hs]


# class MetaLearner(nn.Module):

#     def __init__(self, input_size, hidden_size, n_learner_params):
#         super(MetaLearner, self).__init__()
#         """Args:
#             input_size (int): for the first LSTM layer, default = 4
#             hidden_size (int): for the first LSTM layer, default = 20
#             n_learner_params (int): number of learner's parameters
#         """
#         self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
#         self.metalstm = MetaLSTMCell(input_size=hidden_size, hidden_size=1, n_learner_params=n_learner_params)
#         self.value = Value(571, 571)
#         self.key = Key(571, 20)
#         self.query = Query(571, 20)
#         self.attend = nn.Softmax(dim = -1)
#         self.linear = nn.Linear(in_features=4, out_features=20)

#     def forward(self, inputs, hs=None):
#         """Args:
#             inputs = [loss, grad_prep, grad]
#                 loss (torch.Tensor of size [1, 2])
#                 grad_prep (torch.Tensor of size [n_learner_params, 2])
#                 grad (torch.Tensor of size [n_learner_params])

#             hs = [(lstm_hn, lstm_cn), [metalstm_fn, metalstm_in, metalstm_cn]]
#         """

#         loss, grad_prep, grad = inputs
# #         print(loss)
#         loss = loss.expand_as(grad_prep)
#         inputs = torch.cat((loss, grad_prep), 1)   # [n_learner_params, 4

#         if hs is None:
#             hs = [None, None]

#         lstmhx, lstmcx = self.lstm(inputs, hs[0])
# #         print(inputs.shape)
#         hx = torch.reshape(lstmhx, (748,571))
# #         print(hx[0][0],hx[1][0],hx[2][0])
#         v_h = self.value(hx)
#         k_h = self.key(hx)
#         q_h = self.query(hx)
#         dots = torch.matmul(q_h, k_h.transpose(-1, -2)) * 571 ** -0.5
# #         print(dots[0][0],dots[1][0],dots[2][0])
# #         attn = self.attend(dots)
# #         print(attn)
#         out = torch.matmul(dots, v_h)
# #         print(hx[0][0],out[0][0])
#         attention_h = hx + out
# #         print("att:",attention_h.shape)
# #         att = torch.flatten(attention_h, 0).unsqueeze(1)
#         att = torch.reshape(attention_h, (106777,4))

# #         print(att.shape)
#         flat_learner_unsqzd, metalstm_hs = self.metalstm([lstmhx, grad], hs[1])
# #         print(len(metalstm_hs[0]))
#         return flat_learner_unsqzd.squeeze(), [(lstmhx, lstmcx), metalstm_hs]





# class MetaLearner(nn.Module):

#     def __init__(self, input_size, hidden_size, n_learner_params):
#         super(MetaLearner, self).__init__()
#         """Args:
#             input_size (int): for the first LSTM layer, default = 4
#             hidden_size (int): for the first LSTM layer, default = 20
#             n_learner_params (int): number of learner's parameters
#         """
#         self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
#         self.metalstm = MetaLSTMCell(input_size=hidden_size, hidden_size=1, n_learner_params=n_learner_params)
#         self.value = Value(132, 132)
#         self.key = Key(132, 20)
#         self.query = Query(132, 20)
#         self.attend = nn.Softmax(dim = -1)
#         self.linear = nn.Linear(in_features=4, out_features=20)

#     def forward(self, inputs, hs=None):
#         """Args:
#             inputs = [loss, grad_prep, grad]
#                 loss (torch.Tensor of size [1, 2])
#                 grad_prep (torch.Tensor of size [n_learner_params, 2])
#                 grad (torch.Tensor of size [n_learner_params])

#             hs = [(lstm_hn, lstm_cn), [metalstm_fn, metalstm_in, metalstm_cn]]
#         """

#         loss, grad_prep, grad = inputs
# #         print(loss)
#         loss = loss.expand_as(grad_prep)
#         inputs = torch.cat((loss, grad_prep), 1)   # [n_learner_params, 4

#         if hs is None:
#             hs = [None, None]

#         lstmhx, lstmcx = self.lstm(inputs, hs[0])
# # #         print(inputs.shape)
#         hx = torch.reshape(lstmhx, (997,132))
# #         print(hx[0][0],hx[1][0],hx[2][0])
#         v_h = self.value(hx)
#         k_h = self.key(hx)
#         q_h = self.query(hx)
#         dots = torch.matmul(q_h, k_h.transpose(-1, -2)) * 132 ** -0.5
# #         print(dots[0][0],dots[1][0],dots[2][0])
# #         attn = self.attend(dots)
# #         print(attn)
#         out = torch.matmul(dots, v_h)
# #         print(hx[0][0],out[0][0])
#         attention_h = hx + out
# #         print("att:",attention_h.shape)
# #         att = torch.flatten(attention_h, 0).unsqueeze(1)
#         att = torch.reshape(attention_h, (32901,4))

# #         print(att.shape)
#         flat_learner_unsqzd, metalstm_hs = self.metalstm([att, grad], hs[1])
# #         print(len(metalstm_hs[0]))
#         return flat_learner_unsqzd.squeeze(), [(lstmhx, lstmcx), metalstm_hs]


class MetaLearner(nn.Module):

    def __init__(self, input_size, hidden_size, n_learner_params):
        super(MetaLearner, self).__init__()
        """Args:
            input_size (int): for the first LSTM layer, default = 4
            hidden_size (int): for the first LSTM layer, default = 20
            n_learner_params (int): number of learner's parameters
        """
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.metalstm = MetaLSTMCell(input_size=hidden_size, hidden_size=1, n_learner_params=n_learner_params)
        self.value = Value(948, 948)
        self.key = Key(948, 20)
        self.query = Query(948, 20)
        self.attend = nn.Softmax(dim = -1)
        self.linear = nn.Linear(in_features=4, out_features=20)

    def forward(self, inputs, hs=None):
        """Args:
            inputs = [loss, grad_prep, grad]
                loss (torch.Tensor of size [1, 2])
                grad_prep (torch.Tensor of size [n_learner_params, 2])
                grad (torch.Tensor of size [n_learner_params])

            hs = [(lstm_hn, lstm_cn), [metalstm_fn, metalstm_in, metalstm_cn]]
        """

        loss, grad_prep, grad = inputs
#         print(loss)
        loss = loss.expand_as(grad_prep)
        inputs = torch.cat((loss, grad_prep), 1)   # [n_learner_params, 4
        #print(loss.shape,grad_prep.shape,grad.shape)
        if hs is None:
            hs = [None, None]

        lstmhx, lstmcx = self.lstm(inputs, hs[0])

#         print(inputs.shape, lstmhx.shape)
#         breakpoint()
#         print("1",lstmhx.shape,grad.shape)

        hx = torch.reshape(lstmhx, (1217,948))
#         print(hx[0][0],hx[1][0],hx[2][0])
        v_h = self.value(hx)
        k_h = self.key(hx)
        q_h = self.query(hx)
        dots = torch.matmul(q_h, k_h.transpose(-1, -2)) * 948 ** -0.5
#         print(dots[0][0],dots[1][0],dots[2][0])
#         attn = self.attend(dots)
#         print(attn)
        out = torch.matmul(dots, v_h)
#         print(hx[0][0],out[0][0])
        attention_h = hx + out
#         print("att:",attention_h.shape)
#         att = torch.flatten(attention_h, 0).unsqueeze(1)
        att = torch.reshape(attention_h, (288429,4))

#         print(att.shape)
        #print(lstmhx[0][0],lstmhx[0][1],lstmhx[0][2],lstmhx[0][3])
        #print(lstmhx.shape,grad.shape)
        # [106777,4],[106777,1]
        
        #print(hs[-1][0].shape)


        flat_learner_unsqzd, metalstm_hs = self.metalstm([att, grad], hs[1])

        return flat_learner_unsqzd.squeeze(), [(lstmhx, lstmcx), metalstm_hs]