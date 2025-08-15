import numpy as np 
import torch 
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Ma_Rb_conv(object):
    def __init__(self, critic_dim, actor_dim, action_dim, num_agents,scan_size, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state_dim = critic_dim
        self.actor_dim = actor_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.scan_size = scan_size
        self.state = np.zeros((max_size, critic_dim))
        self.next_state = np.zeros((max_size, critic_dim))
        self.reward = np.zeros((max_size,num_agents))
        self.not_done = np.zeros((max_size,num_agents))


        self.init_actor_memory()


    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []
        self.scan = []
        self.next_scan = []

        for i in range(self.num_agents):
            self.scan.append(np.zeros((self.max_size, self.scan_size)))
            self.next_scan.append(np.zeros((self.max_size, self.scan_size)))
            self.actor_state_memory.append(
                            np.zeros((self.max_size, self.actor_dim)))
            self.actor_new_state_memory.append(
                            np.zeros((self.max_size, self.actor_dim)))
            self.actor_action_memory.append(
                            np.zeros((self.max_size, self.action_dim)))

    
    def add(self, state_critic, state_actor, scan, next_scan, action, next_state_critic, next_state_actor, reward, done):
        for agent_idx in range(self.num_agents):
            self.actor_state_memory[agent_idx][self.ptr ] = state_actor[agent_idx]
            self.actor_new_state_memory[agent_idx][self.ptr ] = next_state_actor[agent_idx]
            self.actor_action_memory[agent_idx][self.ptr ] = action[agent_idx]
            self.scan[agent_idx][self.ptr ] = scan[agent_idx]
            self.next_scan[agent_idx][self.ptr ] = next_scan[agent_idx]

        self.state[self.ptr] = state_critic
        self.next_state[self.ptr] = next_state_critic
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1 - done

        self.ptr = (self.ptr +1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def len(self):
        return self.size

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        actor_states = []
        actor_new_states = []
        actions = []
        scans = []
        next_scans = []
        for agent_idx in range(self.num_agents):
            actor_states.append(self.actor_state_memory[agent_idx][ind])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][ind])
            actions.append(self.actor_action_memory[agent_idx][ind])
            scans.append(self.scan[agent_idx][ind])
            next_scans.append(self.next_scan[agent_idx][ind])
        
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device),
            torch.FloatTensor(np.asarray(actor_states)).to(device),
            torch.FloatTensor(np.asarray(actor_new_states)).to(device),
            torch.FloatTensor(np.asarray(actions)).to(device),
            torch.FloatTensor(np.asarray(scans)).to(device),
            torch.FloatTensor(np.asarray(next_scans)).to(device)
        )
#***********************************************************************************
class TargetNet:
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict)

    def alpha_sync(self, alpha):
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k,v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1- alpha) * v
        self.target_model.load_state_dict(tgt_state)


