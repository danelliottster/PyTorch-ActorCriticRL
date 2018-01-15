from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os, argparse, tempfile, cPickle
import psutil
import gc

import train
import buffer

ver = "1.0.0"


parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--noActorTarget", action="store_true", default=False)
parser.add_argument("--noCriticTarget", action="store_true", default=False)
parser.add_argument("--N_e", type=int, default=1)
parser.add_argument("--combineMedian", action="store_true", default=False)
parser.add_argument("--evalEvery", type=int, default=25)
parser.add_argument("--maxEpisodes", type=int, default=5000)
parser.add_argument("--exploreEpLength", type=int, default=1600)
parser.add_argument("--showEvals", action="store_true", default=False)
parser.add_argument("--saveDir", default=None)
parser.add_argument("--savePrefix", default=None)
parser.add_argument("--rerunNum", type=int, default=None)
args = parser.parse_args()
print args

env = gym.make('BipedalWalker-v2')
# env = gym.make('Pendulum-v0')

MAX_BUFFER = 1000000
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print ' State Dimensions :- ', S_DIM
print ' Action Dimensions :- ', A_DIM
print ' Action Max :- ', A_MAX

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
trainer_indirect = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

evaluation_reward_hist = []; evaluation_distance_hist = [];
evaluation_reward_hist_indirect = []; evaluation_distance_hist_indirect = [];
all_reward_hist = []
directEval_p = True
_ep = 0
while _ep < args.maxEpisodes:
	evalEpisode_p = (_ep % args.evalEvery == 0)
	observation = env.reset()
        episodeR = 0
        t = 0
	while(True):            # loop until we get done signal from environment
                if not evalEpisode_p and t > args.exploreEpLength:
                        break
                if evalEpisode_p and args.showEvals:
                        env.render()
		state = np.float32(observation)

		if evalEpisode_p and directEval_p:
			# evaluation episode, use exploitation policy here
			action = trainer.get_exploitation_action(state)
		elif evalEpisode_p and not directEval_p:
			# evaluation episode, use exploitation policy here
			action = trainer_indirect.get_exploitation_action(state)
		else:
			# get action based on observation, use exploration policy here
			action = trainer.get_exploration_action(state)

		new_observation, reward, done, info = env.step(np.squeeze(action))
                episodeR += reward

		if done:
			new_state = None
		else:
			new_state = np.float32(new_observation)
			# push this exp in ram unless we are doing the indirect eval
                        if directEval_p:
			        ram.add(state, action, reward, new_state)

		observation = new_observation

		if done:
			break
                t += 1

        
	# perform optimization
        for i in range(100):
                trainer.optimize()
                trainer_indirect.optimize()

        if evalEpisode_p and directEval_p:
                evaluation_reward_hist += [(_ep,episodeR)]
                evaluation_distance_hist += [env.env.hull.position[0]]
        elif evalEpisode_p and not directEval_p:
                evaluation_reward_hist_indirect += [(_ep,episodeR)]
                evaluation_distance_hist_indirect += [env.env.hull.position[0]]

        all_reward_hist += [(_ep,episodeR)]
        print 'EPISODE :- ', _ep, t, evalEpisode_p, directEval_p, episodeR, env.env.hull.position[0]
                
	# check memory consumption and clear memory
	gc.collect()

        # hack to force it to do to episodes for an eval episode
        if evalEpisode_p and directEval_p:
                _ep -= 1
                directEval_p = False
        elif not directEval_p:
                directEval_p = True
        _ep += 1

print 'Completed episodes'
print all_reward_hist
print evaluation_reward_hist
print evaluation_distance_hist

if args.savePrefix and args.saveDir:
        saveEvalFile = tempfile.NamedTemporaryFile(mode="w",delete=False,
                                                   dir=args.saveDir,prefix=args.savePrefix,
                                                   suffix=".outfile")
        cPickle.dump(args,saveEvalFile)
        cPickle.dump(evaluation_reward_hist,saveEvalFile)
        cPickle.dump(evaluation_distance_hist, saveEvalFile)
        cPickle.dump(evaluation_reward_hist_indirect,saveEvalFile)
        cPickle.dump(evaluation_distance_hist_indirect, saveEvalFile)
        cPickle.dump(all_reward_hist, saveEvalFile)
        cPickle.dump(ver, saveEvalFile)
        saveEvalFile.close()
