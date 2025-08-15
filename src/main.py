#!/usr/bin/env python

import torch
from env_coop import Env
import rospy
import errno
import os
import numpy as np
import random
import sys 
import copy
import torch
from utilis import Ma_Rb_conv as Replay_buffer
from SAC_ATT import SAC
from tensorboardX import SummaryWriter
import datetime

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

NUM_AGENTS = 3
ACT_STATE_DIMENSION = 12
STATE_DIMENSION = ACT_STATE_DIMENSION * NUM_AGENTS
ACTION_DIMENSION = 2
ACTION_V_MAX = 0.75  # m/s
ACTION_W_MAX = 0.75 # rad/s

REPLAY_SIZE = 1000000
REPLAY_INITIAL = 2000
MAX_STEPS = 1500
SCAN_SIZE = 40 # no. of rays in scan



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":

    rospy.init_node('att')

    data = []
    
    seed =  random.randint(1,100)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    date = datetime.datetime.now().strftime("%d.%h.%H")
    writer = SummaryWriter(dirPath+'/runs/plot/'+str(seed)+"rewards"+date)
    save_path = os.path.join(dirPath, "runs/plot/"+str(seed)+"Nets"+date)
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    env = Env(ACT_STATE_DIMENSION)
    past_action = np.zeros([NUM_AGENTS,ACTION_DIMENSION])
    past_scans = np.zeros([NUM_AGENTS,SCAN_SIZE])
    replay_buffer = Replay_buffer(STATE_DIMENSION, ACT_STATE_DIMENSION, ACTION_DIMENSION, NUM_AGENTS, SCAN_SIZE)
    policy = SAC(seed, STATE_DIMENSION, ACT_STATE_DIMENSION, ACTION_DIMENSION, ACTION_V_MAX, ACTION_W_MAX, SCAN_SIZE, replay_buffer, discount = 0.99, reward_scale=2)

    
    cntr = 0    
    episode = 0
    best_reward = None
    test_episode = 0
    print(seed)
    train_step = 0

    
    
    while episode <= 4_000:
        
        done = False
        episode += 1
 
        rewards_current_episode = np.zeros(NUM_AGENTS)
        print("Seed: ", str(seed), " agent att episode: ", episode)
        r = rospy.Rate(20) ################ set control HZ
        state_c, state_act, scans = env.reset()
        for step in range(MAX_STEPS):
            state_act = np.float32(state_act)
            scans = np.float32(scans)
            action = policy.select_action(state_act, scans)

            next_state_c, next_state_act, reward, done, scans, form_error, act_norm, cols, \
                goals, cols_episode = env.step(action, past_action)
               
            rewards_current_episode += reward
            next_state_act = np.float32(next_state_act)
            next_state_c = np.float32(next_state_c)
        
            replay_buffer.add(state_c,state_act, scans, past_scans, act_norm, next_state_c, next_state_act, reward, done) 
            state_act = copy.deepcopy(next_state_act)
            state_c = copy.deepcopy(next_state_c)
            past_scans = copy.deepcopy(scans)
            past_action = copy.deepcopy(action)

            if replay_buffer.len() > REPLAY_INITIAL:
                train_step = policy.train()
            
            if done.any() or step == MAX_STEPS-1:
                print("Number of collsions: ", cols)
                writer.add_scalar('average_rewards/episode',np.sum(rewards_current_episode)/NUM_AGENTS, episode)
                writer.add_scalar('form_error/episode',form_error, episode)
                writer.add_scalar('goals/episode', goals, episode)   
                break
            r.sleep()


        if episode % 100 == 0:
            ename = "episode %d.dat" % (episode)
            fename = os.path.join(save_path, ename)
            policy.save(fename)
        #**********************************************************************************
print("agent att finished training")
print(seed)


