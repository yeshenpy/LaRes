import time

import numpy as np
import torch
from models import flatten_mlp, tanh_gaussian_actor
from replay_buffer import replay_buffer
from datetime import datetime
import copy, os
from torch.distributions.normal import Normal
from torch.distributions import Distribution
from rlkit.envs.wrappers import NormalizedBoxEnv
"""
The sac is modified to train the sawyer environment

"""
"""
the tanhnormal distributions from rlkit may not stable

"""
class tanh_normal(Distribution):
    def __init__(self, normal_mean, normal_std, epsilon=1e-6, cuda=False):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.cuda = cuda
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        sample_mean = torch.zeros(self.normal_mean.size(), dtype=torch.float32, device='cuda' if self.cuda else 'cpu')
        sample_std = torch.ones(self.normal_std.size(), dtype=torch.float32, device='cuda' if self.cuda else 'cpu')
        z = (self.normal_mean + self.normal_std * Normal(sample_mean, sample_std).sample())
        z.requires_grad_()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


# record the reward info of the dqn experiments
class reward_recorder:
    def __init__(self, history_length=100):
        self.history_length = history_length
        # the empty buffer to store rewards
        self.buffer = [0.0]
        self._episode_length = 1

    # add rewards
    def add_rewards(self, reward):
        self.buffer[-1] += reward

    # start new episode
    def start_new_episode(self):
        if self.get_length >= self.history_length:
            self.buffer.pop(0)
        # append new one
        self.buffer.append(0.0)
        self._episode_length += 1

    # get length of buffer
    @property
    def get_length(self):
        return len(self.buffer)

    @property
    def mean(self):
        return np.mean(self.buffer)

    # get the length of total episodes
    @property
    def num_episodes(self):
        return self._episode_length



class get_action_info:
    def __init__(self, pis, cuda=False):
        self.mean, self.std = pis
        self.dist = tanh_normal(normal_mean=self.mean, normal_std=self.std, cuda=cuda)

    # select actions
    def select_actions(self, exploration=True, reparameterize=True):
        if exploration:
            if reparameterize:
                actions, pretanh = self.dist.rsample(return_pretanh_value=True)
                return actions, pretanh
            else:
                actions = self.dist.sample()
        else:
            actions = torch.tanh(self.mean)
        return actions

    def get_log_prob(self, actions, pre_tanh_value):
        log_prob = self.dist.log_prob(actions, pre_tanh_value=pre_tanh_value)
        return log_prob.sum(dim=1, keepdim=True)
from torch.distributions import Normal, kl_divergence

class sac_agent:
    def __init__(self, model, env_name, env, eval_env, args, our_wandb):
        self.our_wandb= our_wandb
        self.args = args
        self.env = env
        # create eval environment
        self.eval_env = eval_env
        # observation space
        # build up the network that will be used.
        self.total_eval_num = 0
        self.ea_better = 0
        self.pop = []

        for _ in range(args.pop_size):
            self.pop.append(tanh_gaussian_actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.args.hidden_size,self.args.log_std_min, self.args.log_std_max))


        self.qf1 = flatten_mlp(self.env.observation_space.shape[0], self.args.hidden_size, self.env.action_space.shape[0])
        self.qf2 = flatten_mlp(self.env.observation_space.shape[0], self.args.hidden_size, self.env.action_space.shape[0])
        # set the target q functions
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        # build up the policy network

        self.actor_net = tanh_gaussian_actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.args.hidden_size,self.args.log_std_min, self.args.log_std_max)
        # define the optimizer for them
        self.qf1_optim = torch.optim.Adam(self.qf1.parameters(), lr=self.args.q_lr)
        self.qf2_optim = torch.optim.Adam(self.qf2.parameters(), lr=self.args.q_lr)
        # the optimizer for the policy network
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=self.args.p_lr)
        # entorpy target
        self.target_entropy = -1 * self.env.action_space.shape[0]


        self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda' if self.args.cuda else 'cpu')
        # define the optimizer
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.p_lr)
        # define the replay buffer
        self.buffer = replay_buffer(self.args.buffer_size)
        # get the action max
        self.action_max = self.env.action_space.high
        # if use cuda, put tensor onto the gpu
        if self.args.cuda:
            self.actor_net.cuda()
            self.qf1.cuda()
            self.qf2.cuda()
            self.target_qf1.cuda()
            self.target_qf2.cuda()
        # get the reward recorder and success recorder
        self.reward_recorder = reward_recorder(10)
        self.success_recorder = reward_recorder(10)
        # automatically create the folders to save models
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.previous_print = 0
        self.total_steps = 0
        self.ep_num = 0
        self.RL2EA = False

        self.rl_index = None



    def evluate(self, actor_net):
        ep_steps = 0
        ep_reward = 0
        done = False
        obs, _ = self.env.reset()
        while not done:
            ep_steps += 1
            with torch.no_grad():
                obs_tensor = self._get_tensor_inputs(obs)
                pi = actor_net(obs_tensor)
                action = get_action_info(pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                action = action.cpu().numpy()[0]
            # input the actions into the environment
            obs_, reward, done, info = self.env.step(self.action_max * action)
            self.reward_recorder.add_rewards(reward)
            ep_reward += reward
            # store the samples
            self.buffer.add(obs, action, reward, obs_, float(done))
            # reassign the observations
            obs = obs_
            if done:
                self.reward_recorder.start_new_episode()
                self.success_recorder.add_rewards(info['success'])
                self.success_recorder.start_new_episode()

        return ep_reward, ep_steps

    # train the agent
    def learn(self):
        global_timesteps = 0
        # before the official training, do the initial exploration to add episodes into the replay buffer
        self._initial_exploration(exploration_policy=self.args.init_exploration_policy) 
        # reset the environment
        while self.total_steps  < self.args.total_timesteps:

            es_params = self.CEM.ask(self.args.pop_size)
            if not self.RL2EA:
                for i in range(self.args.pop_size):
                    self.pop[i].set_params(es_params[i])
            else:
                for i in range(self.args.pop_size):
                    if i != self.rl_index:
                        self.pop[i].set_params(es_params[i])
                    else:
                        es_params[i] = self.pop[i].get_params()
                        # self.pop[i].actor.set_params(es_params[i])
            self.RL2EA = False

            total_ep_reward = 0
            fitness = np.zeros(len(self.pop))
            for index, actor_net in enumerate(self.pop):
                ep_reward, ep_steps = self.evluate(actor_net)
                total_ep_reward +=ep_steps
                fitness[index] += ep_reward


            print("Fitness", fitness)
            self.CEM.tell(es_params, fitness)

            # start to collect samples
            ep_reward, ep_steps = self.evluate(self.actor_net)
            total_ep_reward += ep_steps

            self.total_steps += total_ep_reward


            best_index = np.argmax(fitness)
            print("best index ", best_index, np.max(fitness), " RL index ", self.rl_index, fitness[self.rl_index])

            if self.total_steps - self.previous_print > self.args.display_interval:
                # start to do the evaluation
                EA_mean_rewards, EA_mean_success = self._evaluate_agent(self.pop[np.argmax(fitness)])

                self.our_wandb.log(
                    {'EA_Rewards': EA_mean_rewards, 'EA_Success': EA_mean_success,'time_steps': self.total_steps})
                print('[{}] Frames: {}, EA Rewards: {:.3f}, Success: {:.3f}'.format(
                        datetime.now(), \
                        self.total_steps, EA_mean_rewards, EA_mean_success))

            #print("current ", self.total_steps ,  self.ep_num , ep_reward, info['success'])
            # after collect the samples, start to update the network
            for _ in range(self.args.update_cycles * total_ep_reward):
                qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss = self._update_newtork(self.pop + [self.actor_net])
                # update the target network
                if global_timesteps % self.args.target_update_interval == 0:
                    self._update_target_network(self.target_qf1, self.qf1)
                    self._update_target_network(self.target_qf2, self.qf2)
                global_timesteps += 1

            # Replace any index different from the new elite
            replace_index = np.argmin(fitness)
            # if replace_index == elite_index:
            #     replace_index = (replace_index + 1) % len(self.pop)
            self.rl_to_evo(self.actor_net, self.pop[replace_index])
            self.RL2EA = True
            self.rl_index = replace_index
            # self.evolver.rl_policy = replace_index
            print('Sync from RL --> Nevo')

            if self.total_steps - self.previous_print >  self.args.display_interval:
                self.previous_print = self.total_steps
                # start to do the evaluation
                mean_rewards, mean_success = self._evaluate_agent(self.actor_net)

                self.total_eval_num +=1.0
                if mean_success > EA_mean_success:
                    self.ea_better +=1.0

                self.our_wandb.log(
                {'EA_better_ratio':self.ea_better/self.total_eval_num ,'Rewards': np.max([mean_rewards, EA_mean_rewards]), 'Success': np.max([EA_mean_success, mean_success]),  'RL_Rewards': mean_rewards, 'RL_Success': mean_success, 'T_Reward': self.reward_recorder.mean, 'Q_loss': qf1_loss ,  'Actor_loss': actor_loss, 'Alpha_loss':alpha_loss, 'Alpha':alpha, 'time_steps': self.total_steps })
                print('[{}] Frames: {}, RL ewards: {:.3f}, Success: {:.3f}, T_Reward: {:.3f}, QF1: {:.3f}, QF2: {:.3f}, AL: {:.3f}, Alpha: {:.3f}, AlphaL: {:.3f}'.format(datetime.now(), \
                        self.total_steps , mean_rewards, mean_success, self.reward_recorder.mean, qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss))

                torch.save(self.actor_net.state_dict(), self.model_path + '/model.pt' if self.args.random_init else self.model_path + '/fixed_model.pt')
                if mean_success == 1:
                    torch.save(self.actor_net.state_dict(), self.model_path + '/best_model.pt' if self.args.random_init else self.model_path + '/fixed_best_model.pt')

    def rl_to_evo(self, rl_agent, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_agent.parameters()):
            target_param.data.copy_(param.data)


    def _initial_exploration_and_store(self, bufer, reward_function_pop, reward_input_name_pop, exploration_policy='gaussian'):
        # get the action information of the environment
        obs,_ = self.env.reset()

        store = []
        for _ in range(self.args.init_exploration_steps):
            if exploration_policy == 'uniform':
                action = np.random.uniform(-1, 1, (self.env.action_space.shape[0], ))
            elif exploration_policy == 'gaussian':
                # the sac does not need normalize?
                with torch.no_grad():
                    obs_tensor = self._get_tensor_inputs(obs)
                    # generate the policy
                    pi = self.actor_net(obs_tensor)
                    action = get_action_info(pi).select_actions(reparameterize=False)
                    action = action.cpu().numpy()[0]
            # input the action input the environment
            obs_, reward, done, _ = self.env.step(self.action_max * action)
            org_info = self.env._env.get_dict()

            other_rewards = []
            for index_index, rf in enumerate(reward_function_pop):

                try:
                    input = []
                    for name in reward_input_name_pop[index_index]:
                        input.append(org_info[name])
                    other_rewards.append(rf(*input)[0])
                except Exception as e:
                    print("Error in reward function", e)
                    other_rewards.append(0.0)

            other_rewards.append(reward)
            # store the episodes
            bufer.add(org_info, obs, action, other_rewards, obs_, float(done))
            #buffer.add(org_info, obs, action, other_rewards, obs_, float(done))
            obs = obs_
            if done:
                # if done, reset the environment
                obs, _ = self.env.reset()
        print("Initial exploration has been finished!")

    # do the initial exploration by using the uniform policy
    def _initial_exploration(self, exploration_policy='gaussian'):
        # get the action information of the environment
        obs,_ = self.env.reset()

        store = []
        for _ in range(100):
            if exploration_policy == 'uniform':
                action = np.random.uniform(-1, 1, (self.env.action_space.shape[0], ))
            elif exploration_policy == 'gaussian':
                # the sac does not need normalize?
                with torch.no_grad():
                    obs_tensor = self._get_tensor_inputs(obs)
                    # generate the policy
                    pi = self.actor_net(obs_tensor)
                    action = get_action_info(pi).select_actions(reparameterize=False)
                    action = action.cpu().numpy()[0]
            # input the action input the environment
            obs_, reward, done, _ = self.env.step(self.action_max * action)
            store.append(self.env._env.get_dict())

            # store the episodes
            #self.buffer.add(obs, action, reward, obs_, float(done))
            obs = obs_
            if done:
                # if done, reset the environment
                obs, _ = self.env.reset()
        #print("Initial exploration has been finished!")
        return store

    # get tensors
    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(0)
        return obs_tensor

    def compute_kl(self, pi_pis, elite_pis):
        mean_pi, std_pi = pi_pis
        mean_elite, std_elite = elite_pis[0].detach(), elite_pis[1].detach()

        # 创建两个正态分布
        dist_pi = Normal(mean_pi, std_pi)
        dist_elite = Normal(mean_elite, std_elite)

        # 计算 KL 散度
        kl = kl_divergence(dist_pi, dist_elite)  # 形状 [batch_size, action_dim]
        kl = kl.sum(dim=1, keepdim=True)  # 形状 [batch_size, 1]
        return kl

    def compute_l2_distance(self, net1, net2):
        l2_distance = 0.0
        for param1, param2 in zip(net1.parameters(), net2.parameters()):
            l2_distance += torch.sum((param1 - param2) ** 2)
        return l2_distance

    # update the network
    def _update_newtork(self, All_buffer, agent_index, elite_index,  dynamic_weight, use_dynamic = False, constraint_kl=False, temp_weight=0.0, elite_net = None, elite_q1 = None, elite_q2 = None):
        # smaple batch of samples from the replay buffer

        t1 = time.time()
        # if use_dynamic:
        #     obses, actions, rewards, obses_, dones = All_buffer.sample_with_elite(self.args.batch_size, agent_index, elite_index,  dynamic_weight)
        # else:
        obses, actions, rewards, obses_, dones = All_buffer.sample(self.args.batch_size, agent_index)
        # preprocessing the data into the tensors, will support GPU later
        obses = torch.tensor(obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        actions = torch.tensor(actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        rewards = torch.tensor(rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        obses_ = torch.tensor(obses_, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - dones, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        t2 = time.time()
        # start to update the actor network
        actor_fau = 0.0
        q1_fau = 0.0
        q2_fau = 0.0
        t3 = time.time()
        pis = self.actor_net(obses)
        actions_info = get_action_info(pis, cuda=self.args.cuda)
        actions_, pre_tanh_value = actions_info.select_actions(reparameterize=True)
        log_prob = actions_info.get_log_prob(actions_, pre_tanh_value)
        t4 = time.time()
        if constraint_kl:
            kl_loss = self.compute_l2_distance(self.actor_net, elite_net)
            kl_return = kl_loss.item()
        else:
            kl_loss = 0.0
            kl_return = 0.0

        t5 = time.time()
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        # use the automatically tuning
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        t6 = time.time()
        # get the param
        alpha = self.log_alpha.exp()
        # get the q_value for new actions
        q_actions_ = torch.min(self.qf1(obses, actions_), self.qf2(obses, actions_))
        actor_loss = (alpha * log_prob - q_actions_).mean()
        factor_weight = 1.0 #(torch.abs(actor_loss)/ (kl_loss + 1e-20)).detach()

        total_actor_loss =  actor_loss  + factor_weight *kl_loss

        t7 = time.time()
        # q value function loss
        q1_value = self.qf1(obses, actions)
        q2_value = self.qf2(obses, actions)
        with torch.no_grad():

            pis_next = self.actor_net(obses_)
            actions_info_next = get_action_info(pis_next, cuda=self.args.cuda)
            actions_next_, pre_tanh_value_next = actions_info_next.select_actions(reparameterize=True)
            log_prob_next = actions_info_next.get_log_prob(actions_next_, pre_tanh_value_next)
            target_q_value_next = torch.min(self.target_qf1(obses_, actions_next_), self.target_qf2(obses_, actions_next_)) - alpha * log_prob_next
            target_q_value = self.args.reward_scale * rewards + inverse_dones * self.args.gamma * target_q_value_next
        t8 = time.time()
        if constraint_kl:
            q1_kl_loss = self.compute_l2_distance(self.qf1, elite_q1)
            q2_kl_loss = self.compute_l2_distance(self.qf2, elite_q2)
            q1_kl_loss_return = q1_kl_loss.item()
            q2_kl_loss_return = q2_kl_loss.item()
        else:
            q1_kl_loss = 0.0
            q2_kl_loss = 0.0
            q1_kl_loss_return = 0.0
            q2_kl_loss_return = 0.0
        t9 = time.time()
        qf1_loss = (q1_value - target_q_value).pow(2).mean() +factor_weight* q1_kl_loss
        qf2_loss = (q2_value - target_q_value).pow(2).mean() +factor_weight* q2_kl_loss

        self.qf1_optim.zero_grad()
        qf1_loss.backward()
        self.qf1_optim.step()
        t10 = time.time()
        # qf2
        self.qf2_optim.zero_grad()
        qf2_loss.backward()
        self.qf2_optim.step()
        t11 = time.time()
        # policy loss

        self.actor_optim.zero_grad()
        total_actor_loss.backward()
        self.actor_optim.step()
        t12 = time.time()
        time_cost_list =[t12-t11, t11-t10, t10-t9, t9-t8, t8-t7, t7-t6, t6-t5, t5-t4, t4-t3, t3-t2, t2-t1]
        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha.item(), alpha_loss.item(), actor_fau, q1_fau, q2_fau, kl_return, q1_kl_loss_return, q2_kl_loss_return, time_cost_list
    
    # update the target network
    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    # evaluate the agent
    def _evaluate_agent(self, actor_net, reward_function=None, reward_input_name=None):
        total_reward = 0
        total_success = 0
        total_own_reward = 0
        all_infos = []
        for _ in range(self.args.eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0 
            success_flag = False
            while True:
                with torch.no_grad():
                    obs_tensor = self._get_tensor_inputs(obs)

                    pi = actor_net(obs_tensor)
                    action = get_action_info(pi, cuda=self.args.cuda).select_actions(exploration=False, reparameterize=False)
                    action = action.detach().cpu().numpy()[0]

                    if np.isnan(action).any() or np.isinf(action).any():
                        print(pi)
                        print(action)
                        print("Action contains NaN values.")
                        return None, None ,None, None
                # input the action into the environment
                obs_, reward, done, info = self.eval_env.step(self.action_max * action)

                if reward_function is not None:
                    org_info = self.eval_env._env.get_dict()
                    input = []
                    for name in reward_input_name:
                        input.append(org_info[name])
                    total_own_reward +=reward_function(*input)[0]

                episode_reward += reward
                success_flag = success_flag or info['success']
                if done:
                    break
                obs = obs_
            all_infos.append(copy.deepcopy(info))
            total_reward += episode_reward
            total_success += 1 if success_flag else 0

        return total_reward / self.args.eval_episodes, total_success / self.args.eval_episodes, all_infos, total_own_reward/self.args.eval_episodes
