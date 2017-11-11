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
parser.add_argument("--evalEvery", type=int, default=25)
parser.add_argument("--maxEpisodes", type=int, default=5000)
# parser.add_argument("--exploreEpLength", type=int, default=1000)
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
if args.noActorTarget and not args.noCriticTarget:
        trainer = train.Trainer_NoActorTarget(S_DIM, A_DIM, A_MAX, ram)
elif not args.noActorTarget and args.noCriticTarget:
        trainer = train.Trainer_NoCriticTarget(S_DIM, A_DIM, A_MAX, ram)
elif args.noActorTarget and args.noCriticTarget:
        trainer = train.Trainer_NoTargetNetworks(S_DIM, A_DIM, A_MAX, ram)
else:
        trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

evaluation_reward_hist = []; evaluation_distance_hist = [];
all_reward_hist = []
for _ep in range(args.maxEpisodes):
	evalEpisode_p = (_ep % args.evalEvery == 0)

	observation = env.reset()
        episodeR = 0
        # TODO: original code had a max of t=1000.  That might not be a bad idea for exploration episodes
        t = 0
	while(True):            # loop until we get done signal from environment
                # if not evalEpisode_p and t > args.exploreEpLength:
                #         break
                if evalEpisode_p and args.showEvals:
                        env.render()
		state = np.float32(observation)

		if evalEpisode_p:
			# evaluation episode, use exploitation policy here
			action = trainer.get_exploitation_action(state)
		else:
			# get action based on observation, use exploration policy here
			action = trainer.get_exploration_action(state)

		new_observation, reward, done, info = env.step(np.squeeze(action))
                episodeR += reward

		if done:
			new_state = None
		else:
			new_state = np.float32(new_observation)
			# push this exp in ram
			ram.add(state, action, reward, new_state)

		observation = new_observation

		# perform optimization
		trainer.optimize()
		if done:
			break
                t += 1


        if evalEpisode_p:
                evaluation_reward_hist += [(_ep,episodeR)]
                evaluation_distance_hist += [env.env.hull.position[0]]
        all_reward_hist += [(_ep,episodeR)]
        print 'EPISODE :- ', _ep, t, evalEpisode_p, episodeR, env.env.hull.position[0]
                
	# check memory consumption and clear memory
	gc.collect()


print 'Completed episodes'
print all_reward_hist
print evaluation_reward_hist

if args.savePrefix and args.saveDir:
        saveEvalFile = tempfile.NamedTemporaryFile(mode="w",delete=False,
                                                   dir=args.saveDir,prefix=args.savePrefix,
                                                   suffix=".outfile")
        cPickle.dump(args,saveEvalFile)
        cPickle.dump(evaluation_reward_hist,saveEvalFile)
        cPickle.dump(evaluation_distance_hist, saveEvalFile)
        cPickle.dump(all_reward_hist, saveEvalFile)
        cPickle.dump(ver, saveEvalFile)
        saveEvalFile.close()
