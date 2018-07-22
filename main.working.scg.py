# v2.0: moved parameter update function call outside of data collection loop so it runs far fewer times
# v2.1: added flag to run version of gym biped that doesn't penalize for falling
# v2.2: added ability to save weights during training
# v2.2.1: added option to save experience replay memory
# v2.2.1: added option to set experience replay memory size
# v2.2.2: added option to see a graph of evaluation

from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os, argparse, tempfile, cPickle
import psutil
import gc
import matplotlib.pyplot as plt


import train_SCG as train
import buffer

ver = "2.2.1"

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--noActorTarget", action="store_true", default=False)
parser.add_argument("--noCriticTarget", action="store_true", default=False)
parser.add_argument("--N_e", type=int, default=1)
parser.add_argument("--numReplays", type=int, default=5)
parser.add_argument("--scgI_actor", type=int, default=10)
parser.add_argument("--scgI_critic", type=int, default=20)
parser.add_argument("--batchSize", type=int, default=1000)
parser.add_argument("--combineMedian", action="store_true", default=False)
parser.add_argument("--evalEvery", type=int, default=25)
parser.add_argument("--maxEpisodes", type=int, default=1500)
parser.add_argument("--exploreEpLength", type=int, default=1600)
parser.add_argument("--showEvals", action="store_true", default=False)
parser.add_argument("--showGraphs", action="store_true", default=False)
parser.add_argument("--saveDir", default=None)
parser.add_argument("--savePrefix", default=None)
parser.add_argument("--rerunNum", type=int, default=None)
parser.add_argument("--noFallPenalty", action="store_true", default=False)
parser.add_argument("--saveWeightsEvery", type=int, default=0) # 0 means don't save
parser.add_argument("--memorySize", type=int, default=1000000) # 0 means don't save
parser.add_argument("--saveMemory", action="store_true", default=False)
parser.add_argument("--noiseDecay", action="store_true", default=False)
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

epsilon = 1.0             # decay
minEpsilon = 0.1
epsilonDecay = np.exp(np.log(minEpsilon)/float(args.maxEpisodes))

ram = buffer.MemoryBuffer(args.memorySize)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram, N_e = args.N_e)

if args.showGraphs:
        fig_eval = plt.figure()
        axHullAng = fig_eval.add_subplot(7,1,1)
        axHullVel = axHullAng.twinx()
        axJointAngle = fig_eval.add_subplot(7,1,2)
        axJointSpeed1 = fig_eval.add_subplot(7,1,3)
        axJointSpeed2 = fig_eval.add_subplot(7,1,4)
        axAction1 = fig_eval.add_subplot(7,1,5)
        axAction2 = fig_eval.add_subplot(7,1,6)
        axR = fig_eval.add_subplot(7,1,7)
        fig_eval.show()

evaluation_reward_hist = []; evaluation_distance_hist = [];
all_reward_hist = []
saveWeightFiles = []
epLenHist = []
for _ep in range(args.maxEpisodes):
	evalEpisode_p = (_ep % args.evalEvery == 0)
        epStateHist = []; epActionHist = []; epRhist = [];
	observation = env.reset()
        episodeR = 0
        t = 0
        if args.noiseDecay:
                epsilon = max(minEpsilon,epsilon*epsilonDecay)
	while(True):            # loop until we get done signal from environment
                if not evalEpisode_p and t > args.exploreEpLength:
                        break
                if evalEpisode_p and args.showEvals:
                        env.render()
		state = np.float32(observation)
                epStateHist += [np.copy(state)]
		if evalEpisode_p:
			# evaluation episode, use exploitation policy here
			# action = trainer.get_exploitation_action(torch.from_numpy(state).unsqueeze(0)).data.numpy()
			action = trainer.get_exploitation_action(torch.from_numpy(state).unsqueeze(0))
		else:
			# get action based on observation, use exploration policy here
			action = trainer.get_exploration_action(torch.from_numpy(state).unsqueeze(0),
                                                                epsilon)
			# action = trainer.get_exploration_action(torch.from_numpy(state).unsqueeze(0),
                        #                                         epsilon).data.numpy()
                epActionHist += [action]

		new_observation, reward, done, info = env.step(np.squeeze(action))
                episodeR += reward
                epRhist += [reward]

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
        if len(ram.buffer) > 1000:
                for i in range(args.numReplays):
                        trainer.optimize_critics(args.scgI_critic,args.batchSize)
                        for j in range(5):
                                trainer.optimize_actor(100)
                        trainer.update_targets()
                

        if evalEpisode_p:
                evaluation_reward_hist += [(_ep,episodeR)]
                evaluation_distance_hist += [env.env.hull.position[0]]
                if args.showGraphs:
                        axHullAng.clear(); axHullVel.clear(); axJointAngle.clear(); axJointSpeed1.clear(); axJointSpeed2.clear(); axAction1.clear(); axAction2.clear(); axR.clear();
                        axHullAng.plot([esh[0] for esh in epStateHist],"-r")
                        axHullVel.plot([esh[1] for esh in epStateHist],"-b")
                        axJointAngle.plot([esh[4] for esh in epStateHist],"-r") # hip 1
                        axJointAngle.plot([esh[6] for esh in epStateHist],"-b") # knee 1
                        axJointAngle.plot([esh[9] for esh in epStateHist],"-g") # hip 2
                        axJointAngle.plot([esh[11] for esh in epStateHist],"-m") # knee 2
                        axJointSpeed1.plot([esh[5] for esh in epStateHist],"-r") # hip 1
                        axJointSpeed1.plot([esh[7] for esh in epStateHist],"-b") # knee 1
                        axJointSpeed2.plot([esh[10] for esh in epStateHist],"-g") # hip 2
                        axJointSpeed2.plot([esh[12] for esh in epStateHist],"-m") # knee 2
                        axAction1.plot([eah[0,0] for eah in epActionHist],"-r")
                        axAction1.plot([eah[0,1] for eah in epActionHist],"-b")
                        axAction2.plot([eah[0,2] for eah in epActionHist],"-g")
                        axAction2.plot([eah[0,3] for eah in epActionHist],"-m")
                        axR.plot([erh for erh in epRhist], "k")
                        axHullAng.set_xticklabels([]); axHullVel.set_xticklabels([]); axJointAngle.set_xticklabels([]); axJointSpeed1.set_xticklabels([]); axJointSpeed2.set_xticklabels([]); axAction1.set_xticklabels([]); axAction2.set_xticklabels([]);
                        axR.set_xlabel("time")
                        axHullAng.set_ylabel("Hull ang"); axHullVel.set_ylabel("Hull ang vel"); axJointAngle.set_ylabel("joint"); axJointSpeed1.set_ylabel("leg 1 speeds"); axJointSpeed2.set_ylabel("leg 2 speeds"); axAction1.set_ylabel("leg 1 actions"); axAction2.set_ylabel("leg 2 actions"); axR.set_ylabel("R");
                        axJointAngle.set_ylim((-1.2,1.2))
                        fig_eval.canvas.draw()
                        plt.pause(0.2)
        all_reward_hist += [(_ep,episodeR)]
        print 'EPISODE :- ', _ep, t, evalEpisode_p, episodeR, env.env.hull.position[0], epsilon
                
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
