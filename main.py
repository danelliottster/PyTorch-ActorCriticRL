# v2.0: moved parameter update function call outside of data collection loop so it runs far fewer times
# v2.1: added flag to run version of gym biped that doesn't penalize for falling
# v2.2: added ability to save weights during training
# v2.2.1: added option to save experience replay memory
# v2.2.1: added option to set experience replay memory size

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

ver = "2.2.1"

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--noActorTarget", action="store_true", default=False)
parser.add_argument("--noCriticTarget", action="store_true", default=False)
parser.add_argument("--actorCriticEnsemble", action="store_true", default=False)
parser.add_argument("--N_e", type=int, default=1)
parser.add_argument("--numReplays", type=int, default=100)
parser.add_argument("--LR_actor", type=float, default=0.001)
parser.add_argument("--decay_actor", type=float, default=0)
parser.add_argument("--decay_critic", type=float, default=0)
parser.add_argument("--combineMedian", action="store_true", default=False)
parser.add_argument("--evalEvery", type=int, default=25)
parser.add_argument("--maxEpisodes", type=int, default=5000)
parser.add_argument("--exploreEpLength", type=int, default=1600)
parser.add_argument("--showEvals", action="store_true", default=False)
parser.add_argument("--saveDir", default=None)
parser.add_argument("--savePrefix", default=None)
parser.add_argument("--rerunNum", type=int, default=None)
parser.add_argument("--noFallPenalty", action="store_true", default=False)
parser.add_argument("--saveWeightsEvery", type=int, default=0) # 0 means don't save
parser.add_argument("--memorySize", type=int, default=1000000) # 0 means don't save
parser.add_argument("--saveMemory", action="store_true", default=False)
args = parser.parse_args()
print args

if args.noFallPenalty:
        env = gym.make('BipedalWalkerNoFallPenalty-v2')
else:
        env = gym.make('BipedalWalker-v2')
# env = gym.make('Pendulum-v0')

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print ' State Dimensions :- ', S_DIM
print ' Action Dimensions :- ', A_DIM
print ' Action Max :- ', A_MAX

ram = buffer.MemoryBuffer(args.memorySize)
if args.N_e > 1 and args.noCriticTarget:
        trainer = train.Trainer_CriticEnsemble(S_DIM, A_DIM, A_MAX, ram, N_e = args.N_e, LR_critic = 0.001, combineMedian=args.combineMedian)
elif args.N_e > 1 and not args.noCriticTarget and not args.actorCriticEnsemble:
        trainer = train.Trainer_CriticEnsembleWithTargets(S_DIM, A_DIM, A_MAX, ram, N_e = args.N_e,
                                                          combineMedian=args.combineMedian,
                                                          LR_actor=args.LR_actor,
                                                          decay_actor=args.decay_actor,
                                                          decay_critic=args.decay_critic)
elif args.N_e > 1 and args.actorCriticEnsemble:
        trainer = train.Trainer_ActorCriticEnsemble(S_DIM, A_DIM, A_MAX, ram, N_e = args.N_e)
elif args.noActorTarget and not args.noCriticTarget:
        trainer = train.Trainer_NoActorTarget(S_DIM, A_DIM, A_MAX, ram)
elif not args.noActorTarget and args.noCriticTarget:
        trainer = train.Trainer_NoCriticTarget(S_DIM, A_DIM, A_MAX, ram)
elif args.noActorTarget and args.noCriticTarget:
        trainer = train.Trainer_NoTargetNetworks(S_DIM, A_DIM, A_MAX, ram)
else:
        trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

evaluation_reward_hist = []; evaluation_distance_hist = [];
all_reward_hist = []
saveWeightFiles = []
epLenHist = []
for _ep in range(args.maxEpisodes):
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

		if done:
			break
                t += 1
        epLenHist += [t]

	# perform optimization
        for i in range(args.numReplays):
	        trainer.optimize()

        if evalEpisode_p:
                evaluation_reward_hist += [(_ep,episodeR)]
                evaluation_distance_hist += [env.env.hull.position[0]]
        all_reward_hist += [(_ep,episodeR)]
        print 'EPISODE :- ', _ep, t, evalEpisode_p, episodeR, env.env.hull.position[0]
                
	# check memory consumption and clear memory
	gc.collect()

        # write weights to file
        if args.saveWeightsEvery and not _ep % args.saveWeightsEvery:
                saveWeightFiles += [trainer.save_models(args.saveDir+"/"+args.savePrefix)]
                

print 'Completed episodes'
print all_reward_hist
print evaluation_reward_hist
print evaluation_distance_hist

if args.savePrefix and args.saveDir:
        saveEvalFile = tempfile.NamedTemporaryFile(mode="w",delete=False,
                                                   dir=args.saveDir,prefix=args.savePrefix,
                                                   suffix=".outfile")
        expRplyMemFile = None
        if args.saveMemory:
                expRplyMemFile = tempfile.NamedTemporaryFile(mode="w",delete=False,
                                                             dir=args.saveDir,prefix=args.savePrefix,
                                                             suffix="_expRplyMem.outfile")
                cPickle.dump(ram, expRplyMemFile)
                expRplyMemFile.close()
        cPickle.dump(ver, saveEvalFile)
        cPickle.dump(args,saveEvalFile)
        cPickle.dump(evaluation_reward_hist,saveEvalFile)
        cPickle.dump(evaluation_distance_hist, saveEvalFile)
        cPickle.dump(all_reward_hist, saveEvalFile)
        cPickle.dump(epLenHist, saveEvalFile)
        cPickle.dump(saveWeightFiles, saveEvalFile)
        if expRplyMemFile:
                cPickle.dump(expRplyMemFile.name, saveEvalFile)
        else:
                cPickle.dump(None, saveEvalFile)
        saveEvalFile.close()
