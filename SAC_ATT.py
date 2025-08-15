#!/usr/bin/env python
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distr
import torch
import numpy as np
import os, math
from utilis import  TargetNet
from tensorboardX import SummaryWriter
import datetime
from ATT_modules import CentralAttention1 as CentralAttention
# from ATT_modules import CentralAttention2 as CentralAttention



date = datetime.datetime.now().strftime("%d.%h.%H")
dirPath = os.path.dirname(os.path.realpath(__file__))



ACTION_V_MAX = 0.75 # m/s
ACTION_W_MAX = 0.75 # rad/s

LR_ACTS = 3e-4
LR_VALS = 3e-4
LR_ALPHA = 3e-4
INIT_TEMPERATURE = 0.1
HID_SIZE = 256
OUT_FEATURES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#***************************** Actor network
class SACActor(nn.Module):
    def init_weights(self,m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight,a=math.sqrt(5))
            m.bias.data.fill_(0.01)

    
    def __init__(self, obs_size, act_size, scan_size):
        super(SACActor, self).__init__()

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
    
        self.mu = nn.Sequential(
            nn.Linear(obs_size + OUT_FEATURES,HID_SIZE),
            nn.GELU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.GELU(),
            nn.Linear(HID_SIZE, act_size)
        )
        
        self.sigma = nn.Sequential(
            nn.Linear(obs_size + OUT_FEATURES,HID_SIZE),
            nn.GELU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.GELU(),
            nn.Linear(HID_SIZE, act_size),
        )
        
        self.conv.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        self.mu.apply(self.init_weights)
        self.sigma.apply(self.init_weights)


    def conv_forward(self, scan):
        return self.conv(scan.view(scan.shape[0],1,scan.shape[1]))
    
    
    def forward(self, x, scan):
        scanx = self.conv_forward(scan)
        scanx = self.fc(scanx)
        xtot = torch.cat([x,scanx],1)
        mu, sigma = self.mu(xtot) , self.sigma(xtot)
        sigma  = torch.clamp(sigma, min=1e-6, max=1)
        return mu, sigma

    def sample_normal(self, state, scan, images, reparameterize=True):
        mu, sigma = self.forward(state, scan, images)
        probs = distr.Normal(mu, sigma)

        if reparameterize:
            x_t = probs.rsample()
        else:
            x_t = probs.sample()

        act_v = torch.sigmoid(x_t[:, None, 0]) * ACTION_V_MAX
        act_w = torch.tanh(x_t[:, None, 1]) * ACTION_W_MAX
        action = torch.cat([act_v,act_w],1).to(device)
        log_probs = probs.log_prob(x_t)
        log_probs -= torch.log(1-action.pow(2)+(1e-6))
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs, mu, sigma

    #********************************** 
class SAC(object):
    def __init__(self,seed, state_dim, actor_state_dim, action_dim, max_action_v, max_action_w, scan_size, replay_buffer, discount = 0.99, reward_scale=2):
        
        self.writer = SummaryWriter(dirPath+'/runs/plot/'+str(seed)+"/losses_"+date)
        self.batch_size = 256
        self.actor = SACActor(actor_state_dim, action_dim, scan_size).to(device)
        self.log_alpha = torch.tensor(np.log(INIT_TEMPERATURE)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim
        self.actor_state_dim = actor_state_dim
        self.nagents = 3 # modify for more
        self.central_attention = CentralAttention(actor_state_dim, action_dim, no_agents=self.nagents, scan_size=scan_size, hidden_dim=128, attend_heads=4, batch_size=self.batch_size).to(device)
        self.central_attention_opt = optim.Adam(self.central_attention.parameters(), lr=LR_VALS)
        self.sched_att = optim.lr_scheduler.ExponentialLR(self.central_attention_opt, gamma=0.9) 
        self.central_attention_tgt = TargetNet(self.central_attention)
        self.act_opt = optim.Adam(self.actor.parameters(), lr=LR_ACTS)
        self.sched_act = optim.lr_scheduler.ExponentialLR(self.act_opt, gamma=0.9)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)
        self.sched_alpha = optim.lr_scheduler.ExponentialLR(self.log_alpha_optimizer, gamma=0.9)

        self.max_action_v = max_action_v
        self.max_action_w = max_action_w
        self.discount = discount
        self.reward_scale = reward_scale
        self.replay_buffer = replay_buffer
        self.state = []
        self.next_state = []
        self.reward = []
        self.not_done = []
        self.all_actor_state = []
        self.all_actor_new_state = []
        self.all_actor_action = []
        self.train_step = 0
        self.scan_size = scan_size
        self.scans = []
        


    
    def select_action(self, states, scans,images, eval=False):
        states_v = torch.FloatTensor(states).to(device)
        scans_v = torch.FloatTensor(scans).to(device)
        images = torch.FloatTensor(np.array(images)).to(device)
        
        if eval == False:
            actions,_,_,_ = self.actor.sample_normal(states_v, scans_v, images, reparameterize=True)
        else:
            _,_,actions,_= self.actor.sample_normal(states_v, scans_v, images, reparameterize=False)
            act_v = torch.sigmoid(actions[:, None, 0]) * ACTION_V_MAX
            act_w = torch.tanh(actions[:, None, 1]) * ACTION_W_MAX
            actions = torch.cat([act_v,act_w],1)
        actions = actions.detach().cpu().numpy()
        return actions
    #*********************************************
    def train(self):
        self.train_step += 1
        
        self.state, self.next_state, self.reward, self.not_done, self.all_actor_state, self.all_actor_new_state, self.all_actor_action, self.scans, self.next_scans,\
            self.images, self.next_images = self.replay_buffer.sample(self.batch_size)
        
        actions_for_critic = self.all_actor_action.view(-1,2)
        all_actor_new_state_one_pass = self.all_actor_new_state.view(-1,self.actor_state_dim)
        all_actor_state_one_pass = self.all_actor_state.view(-1,self.actor_state_dim)
        scans_one_pass = self.scans.view(-1, self.scan_size)
        next_scans_one_pass = self.next_scans.view(-1,self.scan_size)
        images_one_pass = self.images.view(-1, *self.image_size)
        next_images_one_pass = self.next_images.view(-1,*self.image_size)
        
        """
        Policy and Alpha Loss
        """
        actions_new, log_probs_new,_, _ = self.actor.sample_normal(all_actor_state_one_pass, scans_one_pass, images_one_pass, reparameterize=True)
        alpha_loss = (self.log_alpha.exp() * (-log_probs_new - self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        q1_val, q2_val = self.central_attention(all_actor_state_one_pass, actions_new, scans_one_pass, images_one_pass)
        min_val = torch.min(q1_val, q2_val)
        actor_loss = ((self.log_alpha.exp().detach()) * log_probs_new - min_val).mean()

        """
        QF Loss
        """
        q1_v, q2_v = self.central_attention(all_actor_state_one_pass, actions_for_critic, scans_one_pass, images_one_pass)#view(-1)
        with torch.no_grad():
            next_actions_smpld, new_log_probs, _,_ = self.actor.sample_normal(all_actor_new_state_one_pass, next_scans_one_pass, next_images_one_pass, reparameterize=False)
            q1, q2 = self.central_attention_tgt.target_model(all_actor_new_state_one_pass, next_actions_smpld, next_scans_one_pass, next_images_one_pass)
            new_log_probs = new_log_probs.view(-1)
            target_q = torch.min(q1, q2).view(-1) - (self.log_alpha.exp().detach()) * new_log_probs
        #************ reshape rewards and not dones
            rewards = torch.cat([self.reward[:,0], self.reward[:,1], self.reward[:,2]],0) 
            not_dones = torch.cat([self.not_done[:,0], self.not_done[:,1], self.not_done[:,2]],0)
            ref_q = self.reward_scale * rewards+ not_dones  * self.discount * target_q
            ref_q_v  = ref_q
        #*********** calculate losses
        q1_loss_v = F.mse_loss(q1_v.view(-1), ref_q_v.detach())
        q2_loss_v = F.mse_loss(q2_v.view(-1), ref_q_v.detach())
        q_loss_v = q1_loss_v + q2_loss_v

        #***************** update networks
        self.act_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.2)
        self.act_opt.step()
        if self.train_step%100 == 0:
            self.writer.add_scalar('actor_losses/train_step',actor_loss, self.train_step)

        self.central_attention_opt.zero_grad(set_to_none=True)
        q_loss_v.backward()
        #self.central_attention.scale_shared_grads()
        torch.nn.utils.clip_grad_norm_(self.central_attention.parameters(),1)
        self.central_attention_opt.step()
        if self.train_step%100 == 0:
            self.writer.add_scalar('q_losses/train_step',q_loss_v, self.train_step)


        self.central_attention_tgt.alpha_sync(alpha=1 - 1e-3)
        # if self.train_step >= 1e6:
        #     if self.train_step % 1e5 == 0:
        #         self.sched_att.step()
        #         self.sched_act.step()
        #         self.sched_alpha.step()  
        
        return self.train_step



            
            
            
    def save(self, filename):
        torch.save(self.central_attention.state_dict(), filename + "central_attention")
        torch.save(self.central_attention_opt.state_dict(), filename + "central_attention_opt")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.act_opt.state_dict(), filename + "_actor_optimizer")

 

    def load(self, filename):
        self.central_attention.load_state_dict(torch.load(filename + "central_attention"))
        self.central_attention_opt.load_state_dict(torch.load(filename + "central_attention_opt"))
        self.central_attention_tgt = TargetNet(self.central_attention)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.act_opt.load_state_dict(torch.load(filename + "_actor_optimizer"))