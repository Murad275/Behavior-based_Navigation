#!/usr/bin/env python
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


HID_SIZE = 256
OUT_FEATURES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#**********************************
class CentralAttention1(nn.Module):

    def init_weights(self,m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def __init__(self, state_size, action_size, no_agents, scan_size, hidden_dim=32, attend_heads=4, batch_size = 256): 
        """
        Inputs:
            state_size, action_size: Size of state and action spaces of agents
            hidden_dim (int): Number of hidden dimensions
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(CentralAttention1, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.state_size = state_size
        self.action_size = action_size
        self.nagents = no_agents
        self.scan_size = scan_size
        self.attend_heads = attend_heads
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size


        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()           
            )
        test = np.ones((1, 1, scan_size))
        with torch.no_grad():
                n_features = self.conv(torch.as_tensor(test).float()).shape[1]


        self.fc = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, OUT_FEATURES)
            )
        
        self.q1 = nn.Sequential(
                nn.Linear(2* hidden_dim , HID_SIZE),
                nn.ReLU(),
                nn.Linear(HID_SIZE, HID_SIZE),
                nn.ReLU(),
                nn.Linear(HID_SIZE, 1)
            )

        self.q2 = nn.Sequential(
                nn.Linear(2* hidden_dim , HID_SIZE),
                nn.ReLU(),
                nn.Linear(HID_SIZE, HID_SIZE),
                nn.ReLU(),
                nn.Linear(HID_SIZE, 1)
            )
        
        self.fc.apply(self.init_weights)
        self.q1.apply(self.init_weights)
        self.q2.apply(self.init_weights)
        indim = self.state_size + self.action_size

         # used to get e1-eN >> 
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(indim + OUT_FEATURES,affine=False),
            nn.Linear(indim + OUT_FEATURES, hidden_dim),
            nn.LeakyReLU()
        )
        self.encoder.apply(self.init_weights)


        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.queries = nn.ModuleList()
        self.value_extractors = nn.ModuleList()

        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim , attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim, attend_dim),
                                                       nn.LeakyReLU()))
            self.key_extractors[i].apply(self.init_weights)
            self.value_extractors[i].apply(self.init_weights)

        for i in range(3*attend_heads):
                self.queries.append(nn.Linear(hidden_dim, attend_dim, bias=False))
                self.queries[i].apply(self.init_weights)

        self.shared_modules = [ self.key_extractors, self.value_extractors, self.q1, self.q2, self.conv, self.fc]
        
    def conv_forward(self, scan):
        conv_out = self.conv(scan.view(scan.shape[0],1,scan.shape[1]))
        ret = self.fc(conv_out)
        return ret

    def forward(self, obs, acts, scan):
        scanned_features = self.conv_forward(scan)
        inps = torch.cat([torch.cat((obs, acts), dim=1), scanned_features],1)
        sa_encodings_v = self.encoder(inps) 
        sa_encodings = sa_encodings_v.reshape(self.nagents, self.batch_size, self.hidden_dim)
        all_heads_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors] 
        all_heads_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        all_heads_query = []
        all_heads_query.append([query(sa_encodings[0]) for query in self.queries[0:4]]) #pass state encodings into  each query head for current agent
        all_heads_query.append([query(sa_encodings[1]) for query in self.queries[4:8]]) #pass state encodings into  each query head for current agent
        all_heads_query.append([query(sa_encodings[2]) for query in self.queries[8:12]]) #pass state encodings into  each query head for current agent
        
        catted_torches = []
        # calculate attention per head
        for ind in range (self.nagents):
            other_all_values = []
            for curr_head_keys, curr_head_values, query in zip(all_heads_keys, all_heads_values, all_heads_query[ind]):
                keys = [k for j, k in enumerate(curr_head_keys) if j != ind] # excluding key of current agent from each key head
                values = [v for j, v in enumerate(curr_head_values) if j != ind] # excluding value of current agent from each value head
                # calculate attention across agents
                attend_logits = torch.matmul(query.view(query.shape[0], 1, -1),
                                            torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) * attend_weights).sum(dim=2) 
                other_all_values.append(other_values)
            catted_torches.append(torch.cat((sa_encodings[ind], *other_all_values), dim=1))
        all_agents_embeddings = torch.cat([catted_torches[0], catted_torches[1], catted_torches[2]],0)
        q1_rtrn = self.q1(all_agents_embeddings)
        q2_rtrn = self.q2(all_agents_embeddings)
        return q1_rtrn, q2_rtrn
#********************************** 
class Attention(nn.Module):
    def __init__(self, num_heads=4, hid_size=HID_SIZE):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hid_size // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(hid_size, hid_size * 3)
        self.proj = nn.Linear(hid_size, hid_size)
        
    def forward(self, x):   
        B, N, C = x.shape 
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)   
        x = self.proj(x)    
        return x
    
#**********************************      
class CentralAttention2(nn.Module):

    def init_weights(self,m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def __init__(self, state_size, action_size, no_agents, scan_size, hidden_dim=256, attend_heads=4, batch_size = 256):

        super(CentralAttention2, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.state_size = state_size
        self.action_size = action_size
        self.nagents = no_agents
        self.scan_size = scan_size
        self.attend_heads = attend_heads
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
            )
        test = np.ones((1, 1, scan_size))
        
        
        
        with torch.no_grad():
            n_features = self.conv(torch.as_tensor(test).float()).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, OUT_FEATURES)
            )

        
        self.q1 = nn.Sequential(
                nn.Linear( hidden_dim , HID_SIZE),
                nn.GELU(),
                nn.Linear(HID_SIZE, HID_SIZE),
                nn.GELU(),
                nn.Linear(HID_SIZE, 1)
            )

        self.q2 = nn.Sequential(
                nn.Linear( hidden_dim , HID_SIZE),
                nn.GELU(),
                nn.Linear(HID_SIZE, HID_SIZE),
                nn.GELU(),
                nn.Linear(HID_SIZE, 1)
            )
        self.conv.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        self.q1.apply(self.init_weights)
        self.q2.apply(self.init_weights)
        indim = self.state_size + self.action_size
        
        self.embedder = nn.Linear(indim + OUT_FEATURES, hidden_dim)
        self.embedder.apply(self.init_weights)
        
        self.parameterize = nn.Sequential(
                nn.Linear(hidden_dim, HID_SIZE),
                nn.GELU(),
                nn.Linear(HID_SIZE, hidden_dim))
        
        self.parameterize.apply(self.init_weights)
                  

        self.att_layers = Attention(num_heads=attend_heads, hid_size=hidden_dim) # the attention mechanism


    def conv_forward(self, scan):
        conv_out = self.conv(scan.view(scan.shape[0],1,scan.shape[1]))
        ret = self.fc(conv_out)
        return ret


    def forward(self, obs, acts, scan):
        scan_process = self.conv_forward(scan)
        inps = torch.cat([torch.cat((obs, acts), dim=1), scan_process],1)
        inps = inps.view(self.nagents, self.batch_size, -1)
        inps = inps.permute(1,0,2)
        x = self.embedder(inps)
        x = x + self.att_layers(x+ self.parameterize(x)) 
        x = x.permute(1,0,2).reshape(self.batch_size*self.nagents, self.hidden_dim)
        q1_rtrn = self.q1(x)
        q2_rtrn = self.q2(x)
        return q1_rtrn, q2_rtrn
   