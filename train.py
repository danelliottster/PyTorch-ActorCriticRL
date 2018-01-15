from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math, uuid

import utils
import model

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001


class Trainer:

	def __init__(self, state_dim, action_dim, action_lim, ram):
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:param ram: replay memory buffer object
		:return:
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim
		self.ram = ram
		self.iter = 0
		self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

		self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

		self.critic = model.Critic(self.state_dim, self.action_dim)
		self.target_critic = model.Critic(self.state_dim, self.action_dim)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)

        def update_targets(self):
		utils.soft_update(self.target_actor, self.actor, TAU)
		utils.soft_update(self.target_critic, self.critic, TAU)

	def get_exploitation_action(self, state):
		"""
		gets the action from target actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		state = Variable(torch.from_numpy(state).unsqueeze(0))
		action = self.target_actor.forward(state).detach()
		return action.data.numpy()

	def get_exploration_action(self, state):
		"""
		gets the action from actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		state = Variable(torch.from_numpy(state).unsqueeze(0))
		action = self.actor.forward(state).detach()
		new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
		return new_action

	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
		s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)

		s1 = Variable(torch.from_numpy(s1))
		a1 = Variable(torch.from_numpy(a1))
		r1 = Variable(torch.from_numpy(r1))
		s2 = Variable(torch.from_numpy(s2))

		# ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation
		a2 = self.target_actor.forward(s2).detach()
		next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
		# y_exp = r + gamma*Q'( s2, pi'(s2))
		y_expected = r1 + GAMMA*next_val
		# y_pred = Q( s1, a1)
		y_predicted = torch.squeeze(self.critic.forward(s1, a1.squeeze(1)))
		# compute critic loss, and update the critic
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		self.critic_optimizer.zero_grad()
		loss_critic.backward()
		self.critic_optimizer.step()

		# ---------------------- optimize actor ----------------------
		pred_a1 = self.actor.forward(s1)
		loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1))
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

                self.update_targets()

		# if self.iter % 100 == 0:
		# 	print 'Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.numpy(),\
		# 		' Loss_critic :- ', loss_critic.data.numpy()
		# self.iter += 1

	def save_models(self, fileNamePrefix):
		"""
		saves the target critic models
                the actor target model has the same filename prefix as the first critic target.
		:param episode_count: the count of episodes iterated
		:return: list of file names
		"""
                fileName = "%s_%s"%(fileNamePrefix,uuid.uuid4().hex)
		torch.save(self.target_critic.state_dict(),
                           fileName + '_critic.PyTorch')
		torch.save(self.target_actor.state_dict(),
                           fileName + '_actor.PyTorch')
		print 'Models saved successfully'
                return fileName

	def load_models(self, fileName):
		"""
		loads the target actor and critic models, and copies them onto actor and critic models
		:param episode: the count of episodes iterated (used to find the file name)
		:return:
		"""
		self.actor.load_state_dict(torch.load(fileName + '_actor.PyTorch'))
		self.critic.load_state_dict(torch.load(fileName + '_critic.PyTorch'))
		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)
		print 'Models loaded succesfully'

class Trainer_NoActorTarget(Trainer):
        def update_targets(self):
		utils.hard_update(self.target_actor, self.actor)
		utils.soft_update(self.target_critic, self.critic, TAU)

class Trainer_NoCriticTarget(Trainer):
        def update_targets(self):
		utils.soft_update(self.target_actor, self.actor, TAU)
		utils.hard_update(self.target_critic, self.critic)

class Trainer_NoTargetNetworks(Trainer):
        def update_targets(self):
		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)

class Trainer_CriticEnsemble(Trainer):
	def __init__(self, state_dim, action_dim, action_lim, ram, N_e=1, LR_critic=0.001, LR_actor=0.001, combineMedian=False):
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:param ram: replay memory buffer object
		:return:
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim
		self.ram = ram
                self.N_e = N_e
                self.LR_critic = LR_critic
                self.LR_actor = LR_actor
		self.iter = 0
		self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)
                self.combineMedian = combineMedian

		self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

                self.critics = []; self.critic_optimizers = [];
                for nei in range(self.N_e):
                        self.critics += [model.Critic(self.state_dim, self.action_dim)]
                        self.critic_optimizers += [torch.optim.Adam(self.critics[nei].parameters(),self.LR_critic)]

		utils.hard_update(self.target_actor, self.actor)

        def update_targets(self):
		utils.soft_update(self.target_actor, self.actor, TAU)

	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
		# ---------------------- optimize all critics ----------------------
		# Use target actor exploitation policy here for loss evaluation
                for nei in range(self.N_e):
                        s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)
                        s1 = Variable(torch.from_numpy(s1))
                        a1 = Variable(torch.from_numpy(a1))
                        r1 = Variable(torch.from_numpy(r1))
                        s2 = Variable(torch.from_numpy(s2))
                        a2 = self.target_actor.forward(s2).detach()
                        next_val = torch.squeeze(self.critics[nei].forward(s2, a2).detach())
		        # y_exp = r + gamma*Q'( s2, pi'(s2))
		        y_expected = r1 + GAMMA*next_val
		        # y_pred = Q( s1, a1)
		        y_predicted = torch.squeeze(self.critics[nei].forward(s1, a1.squeeze(1)))
		        # compute critic loss, and update the critic
		        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		        self.critic_optimizers[nei].zero_grad()
		        loss_critic.backward()
		        self.critic_optimizers[nei].step()

		# ---------------------- optimize actor ----------------------
		s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)
		s1 = Variable(torch.from_numpy(s1))
		a1 = Variable(torch.from_numpy(a1))
		r1 = Variable(torch.from_numpy(r1))
		s2 = Variable(torch.from_numpy(s2))
		pred_a1 = self.actor.forward(s1)
                member_losses = []
                for nei in range(self.N_e):
                        member_losses += [self.critics[nei].forward(s1,pred_a1)]
                if self.combineMedian:
                        loss_tmp = torch.median(torch.cat(member_losses,dim=1),dim=1)[0]
                else:
                        loss_tmp = torch.mean(torch.cat(member_losses,dim=1),dim=1)
		loss_actor = -1*torch.sum(loss_tmp)
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

                self.update_targets()
        

class Trainer_CriticEnsembleWithTargets(Trainer):
	def __init__(self, state_dim, action_dim, action_lim, ram, N_e=1, LR_critic=0.001, LR_actor=0.001, combineMedian=False, decay_actor=0, decay_critic=0):
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:param ram: replay memory buffer object
		:return:
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim
		self.ram = ram
                self.N_e = N_e
                self.LR_critic = LR_critic
                self.LR_actor = LR_actor
                self.decay_critic = decay_critic
                self.decay_actor = decay_actor
		self.iter = 0
		self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)
                self.combineMedian = combineMedian

		self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                        self.LR_actor,
                                                        weight_decay=self.decay_actor)

                self.critics = []; self.critic_optimizers = []; self.target_critics = [];
                for nei in range(self.N_e):
                        self.critics += [model.Critic(self.state_dim, self.action_dim)]
                        self.critic_optimizers += [torch.optim.Adam(self.critics[nei].parameters(),
                                                                    self.LR_critic,
                                                                    weight_decay=self.decay_critic)]
                        self.target_critics += [model.Critic(self.state_dim, self.action_dim)]

		utils.hard_update(self.target_actor, self.actor)
                for nei in range(self.N_e):
		        utils.hard_update(self.target_critics[nei], self.critics[nei])

        def update_targets(self):
		utils.soft_update(self.target_actor, self.actor, TAU)
                for nei in range(self.N_e):
		        utils.soft_update(self.target_critics[nei], self.critics[nei], TAU)

	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
		# ---------------------- optimize all critics ----------------------
		# Use target actor exploitation policy here for loss evaluation
                for nei in range(self.N_e):
                        s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)
                        s1 = Variable(torch.from_numpy(s1))
                        a1 = Variable(torch.from_numpy(a1))
                        r1 = Variable(torch.from_numpy(r1))
                        s2 = Variable(torch.from_numpy(s2))
                        a2 = self.target_actor.forward(s2).detach()
                        next_val = torch.squeeze(self.target_critics[nei].forward(s2, a2).detach())
		        # y_exp = r + gamma*Q'( s2, pi'(s2))
		        y_expected = r1 + GAMMA*next_val
		        # y_pred = Q( s1, a1)
		        y_predicted = torch.squeeze(self.critics[nei].forward(s1, a1.squeeze(1)))
		        # compute critic loss, and update the critic
		        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		        self.critic_optimizers[nei].zero_grad()
		        loss_critic.backward()
		        self.critic_optimizers[nei].step()

		# ---------------------- optimize actor ----------------------
		s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)
		s1 = Variable(torch.from_numpy(s1))
		a1 = Variable(torch.from_numpy(a1))
		r1 = Variable(torch.from_numpy(r1))
		s2 = Variable(torch.from_numpy(s2))
		pred_a1 = self.actor.forward(s1)
                member_losses = []
                for nei in range(self.N_e):
                        member_losses += [self.critics[nei].forward(s1,pred_a1)]
                if self.combineMedian:
                        loss_tmp = torch.median(torch.cat(member_losses,dim=1),dim=1)[0]
                else:
                        loss_tmp = torch.mean(torch.cat(member_losses,dim=1),dim=1)
		loss_actor = -1*torch.sum(loss_tmp)
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()
                self.update_targets()
        
	def save_models(self, fileNamePrefix):
		"""
		saves the target critic models
                the actor target model has the same filename prefix as the first critic target.
		:param episode_count: the count of episodes iterated
		:return: list of file names
		"""
                fileNameList = []
                for nei in range(self.N_e):
                        fileNameList += ["%s_%s"%(fileNamePrefix,uuid.uuid4().hex)]
		        torch.save(self.target_critics[nei].state_dict(),
                                   fileNameList[nei] + '_critic.PyTorch')
		torch.save(self.target_actor.state_dict(), fileNameList[0]+ '_actor.PyTorch')
		print 'Models saved successfully'
                return fileNameList

	def load_models(self, fileNameList):
		"""
		loads the target actor models and critic model, and copies them onto actor and critic models
		:param episode: the count of episodes iterated (used to find the file name)
		:return:
		"""
                for nei in range(self.N_e):
		        self.critics[nei].load_state_dict(torch.load(fileNameList[nei] + '_critic.PyTorch'))
		        utils.hard_update(self.target_critics[nei], self.critics[nei])
		self.actor.load_state_dict(torch.load(fileNameList[0] + '_actor.PyTorch'))
		utils.hard_update(self.target_actor, self.actor)
		print 'Models loaded succesfully'

class Trainer_ActorCriticEnsemble(Trainer):

	def __init__(self, state_dim, action_dim, action_lim, ram, N_e=1, LR_critic=0.001, LR_actor=0.001, combineMedians=False):
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:param ram: replay memory buffer object
		:return:
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim
		self.ram = ram
		self.iter = 0
		self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)
                self.N_e = N_e
                self.LR_critic = LR_critic
                self.LR_actor = LR_actor
                self.actors = []
                self.target_actors = []
                self.actor_optimizers = []
                self.critics = []
                self.target_critics = []
                self.critic_optimizers = []
                self.combineMedians = combineMedians

                for nei in range(self.N_e):
		        self.actors += [model.Actor(self.state_dim, self.action_dim, self.action_lim)]
		        self.target_actors += [model.Actor(self.state_dim, self.action_dim, self.action_lim)]
		        self.actor_optimizers += [torch.optim.Adam(self.actors[-1].parameters(),LEARNING_RATE)]
		        self.critics += [model.Critic(self.state_dim, self.action_dim)]
		        self.target_critics += [model.Critic(self.state_dim, self.action_dim)]
		        self.critic_optimizers += [torch.optim.Adam(self.critics[nei].parameters(),LEARNING_RATE)]
		        utils.hard_update(self.target_actors[nei], self.actors[nei])
		        utils.hard_update(self.target_critics[nei], self.critics[nei])

        def update_targets(self):
                for nei in range(self.N_e):
		        utils.soft_update(self.target_actors[nei], self.actors[nei], TAU)
		        utils.soft_update(self.target_critics[nei], self.critics[nei], TAU)

	def get_exploitation_action(self, state):
		"""
		gets the action from target actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
                actions = []
		state = Variable(torch.from_numpy(state).unsqueeze(0))
                for nei in range(self.N_e):
		        actions += [self.target_actors[nei].forward(state).detach().data]
                if self.combineMedians:
                        action = torch.median(torch.cat(actions,dim=0),dim=0)
                else:
                        action = torch.mean(torch.cat(actions,dim=0),dim=0)
		return action.numpy()

	def get_exploration_action(self, state):
		"""
		gets the action from actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
                action = self.get_exploitation_action(state)
		new_action = action + (self.noise.sample() * self.action_lim)
		return new_action

	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
                for nei in range(self.N_e):
                        s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)

                        s1 = Variable(torch.from_numpy(s1))
                        a1 = Variable(torch.from_numpy(a1))
                        r1 = Variable(torch.from_numpy(r1))
                        s2 = Variable(torch.from_numpy(s2))

                        # ---------------------- optimize critic ----------------------
                        # Use target actor exploitation policy here for loss evaluation
                        a2 = self.target_actors[nei].forward(s2).detach()
                        next_val = torch.squeeze(self.target_critics[nei].forward(s2, a2).detach())
                        # y_exp = r + gamma*Q'( s2, pi'(s2))
                        y_expected = r1 + GAMMA*next_val
                        # y_pred = Q( s1, a1)
                        y_predicted = torch.squeeze(self.critics[nei].forward(s1, a1.squeeze(1)))
                        # compute critic loss, and update the critic
                        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
                        self.critic_optimizers[nei].zero_grad()
                        loss_critic.backward()
                        self.critic_optimizers[nei].step()

                        # ---------------------- optimize actor ----------------------
                        pred_a1 = self.actors[nei].forward(s1)
                        loss_actor = -1*torch.sum(self.critics[nei].forward(s1, pred_a1))
                        self.actor_optimizers[nei].zero_grad()
                        loss_actor.backward()
                        self.actor_optimizers[nei].step()

                self.update_targets()

	def save_models(self, fileNamePrefix):
		"""
		saves the target actor and critic models
		:param episode_count: the count of episodes iterated
		:return: list of file names
		"""
                fileNameList = []
                for nei in range(self.N_e):
                        fileNameList += ["%s_%s"%(fileNamePrefix,uuid.uuid4().hex)]
		        torch.save(self.target_actor.state_dict(), fileNameList[-1]+ '_actor.PyTorch')
		        torch.save(self.target_critic.state_dict(), fileNameList[-1] + '_critic.PyTorch')
		print 'Models saved successfully'

	def load_models(self, fileNameList):
		"""
		loads the target actor and critic models, and copies them onto actor and critic models
		:param episode: the count of episodes iterated (used to find the file name)
		:return:
		"""
                for nei in range(self.N_e):
		        self.actor.load_state_dict(torch.load(fileNameList[nei] + '_actor.PyTorch'))
		        self.critic.load_state_dict(torch.load(fileNameList[nei] + '_critic.PyTorch'))
		        utils.hard_update(self.target_actor[nei], self.actor[nei])
		        utils.hard_update(self.target_critic[nei], self.critic[nei])
		print 'Models loaded succesfully'

class Trainer_NoActorTarget(Trainer):
        def update_targets(self):
		utils.hard_update(self.target_actor, self.actor)
		utils.soft_update(self.target_critic, self.critic, TAU)

class Trainer_NoCriticTarget(Trainer):
        def update_targets(self):
		utils.soft_update(self.target_actor, self.actor, TAU)
		utils.hard_update(self.target_critic, self.critic)

class Trainer_NoTargetNetworks(Trainer):
        def update_targets(self):
		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)
