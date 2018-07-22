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
import os, argparse, tempfile, cPickle, math
import psutil
import gc
import matplotlib.pyplot as plt


import train
import buffer

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

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
parser.add_argument("--epLength", type=int, default=200)
parser.add_argument("--showEvals", action="store_true", default=False)
parser.add_argument("--showGraphs", action="store_true", default=False)
parser.add_argument("--saveDir", default=None)
parser.add_argument("--savePrefix", default=None)
parser.add_argument("--rerunNum", type=int, default=None)
parser.add_argument("--noFallPenalty", action="store_true", default=False)
parser.add_argument("--saveWeightsEvery", type=int, default=0) # 0 means don't save
parser.add_argument("--memorySize", type=int, default=1000000) # 0 means don't save
parser.add_argument("--saveMemory", action="store_true", default=False)
args = parser.parse_args()
# # args.showEvals = True
# args.showGraphs = True
# args.N_e=1
# args.epLength = 200
# args.numReplays = 25
print args

env = gym.make('Pendulum-v0')
envEval = gym.make('Pendulum_eval-v0')

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print ' State Dimensions :- ', S_DIM
print ' Action Dimensions :- ', A_DIM
print ' Action Max :- ', A_MAX

ram = buffer.MemoryBuffer(args.memorySize)
if args.noCriticTarget:
        trainer = train.Trainer_CriticEnsemble(S_DIM, A_DIM, A_MAX, ram, N_e = args.N_e, LR_critic = 0.001, combineMedian=args.combineMedian)
elif not args.noCriticTarget and not args.actorCriticEnsemble:
        trainer = train.Trainer_CriticEnsembleWithTargets(S_DIM, A_DIM, A_MAX, ram, N_e = args.N_e,
                                                          combineMedian=args.combineMedian,
                                                          LR_actor=args.LR_actor,
                                                          decay_actor=args.decay_actor,
                                                          decay_critic=args.decay_critic,
                                                          actorLayerSizes=[20,20,20],
                                                          criticLayerSizes=[20,20,20,10])
                                                          # actorLayerSizes=[10,10,5],
                                                          # criticLayerSizes=[10,10,10,5])
# elif args.N_e > 1 and args.actorCriticEnsemble:
#         trainer = train.Trainer_ActorCriticEnsemble(S_DIM, A_DIM, A_MAX, ram, N_e = args.N_e)
# elif args.noActorTarget and not args.noCriticTarget:
#         trainer = train.Trainer_NoActorTarget(S_DIM, A_DIM, A_MAX, ram)
# elif not args.noActorTarget and args.noCriticTarget:
#         trainer = train.Trainer_NoCriticTarget(S_DIM, A_DIM, A_MAX, ram)
# elif args.noActorTarget and args.noCriticTarget:
#         trainer = train.Trainer_NoTargetNetworks(S_DIM, A_DIM, A_MAX, ram)
# else:
#         trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

if args.showGraphs:
        fig_eval = plt.figure(0)
        QgridSz = int(math.floor(math.sqrt(args.N_e)))+1
        subRows = 9 + QgridSz; subCols = QgridSz
        axAct = plt.subplot2grid((subRows,subCols), (0,0),colspan=subCols)
        axAng = plt.subplot2grid((subRows,subCols), (1,0),colspan=subCols)
        axAngVel = axAng.twinx()
        axR = plt.subplot2grid((subRows,subCols), (2,0), colspan=subCols)
        axActionSpace = plt.subplot2grid((subRows,subCols), (3, 0), rowspan=3, colspan=subCols)
        Qgrid = []
        for nei in range(args.N_e):
                Qgrid += [plt.subplot2grid( (subRows,subCols), (6+int(float(nei)/float(QgridSz)),nei % QgridSz) )]
        axQmean = plt.subplot2grid((subRows,subCols), (subRows-3, 0), rowspan=3, colspan=subCols)
        fig_eval.show()

evaluation_reward_hist = []; evaluation_distance_hist = [];
all_reward_hist = []
saveWeightFiles = []
epLenHist = []
for _ep in range(args.maxEpisodes):
	evalEpisode_p = (_ep % args.evalEvery == 0)
        epStateHist = []; epActionHist = []; epRhist = [];
        if evalEpisode_p:
	        # observation = np.expand_dims(envEval.reset(),1)
	        observation = envEval.reset()
        else:
	        # observation = env.reset()
	        observation = np.expand_dims(env.reset(),1)
        episodeR = 0
        t = 0
	while(t < args.epLength):            # loop until we get done signal from environment or hit max time steps
                if evalEpisode_p and args.showEvals:
                        envEval.render()
		state = np.float32(observation)
		if evalEpisode_p:
			# evaluation episode, use exploitation policy here
			action = trainer.get_exploitation_action(state.squeeze(1))
                        epStateHist += [np.copy(envEval.env.state)]
                        epActionHist += [action]
		else:
			# get action based on observation, use exploration policy here
			action = trainer.get_exploration_action(state.squeeze(1))

                if evalEpisode_p:
                        new_observation, reward, done, info = envEval.step(action)
                else:
                        new_observation, reward, done, info = env.step(action)
                episodeR += reward
                epRhist += [reward]

		if done:
			new_state = None
		else:
			new_state = np.float32(new_observation)
			# push this exp in ram
                        if not evalEpisode_p:
			        ram.add(state.squeeze(1), action, reward, new_state.squeeze(1))

		observation = new_observation

		if done:
			break
                t += 1

	# perform optimization
        if not evalEpisode_p:
                for i in range(args.numReplays):
	                trainer.optimize()

        if evalEpisode_p:
                evaluation_reward_hist += [(_ep,episodeR)]
                if args.showGraphs:
                        
                        aGrid,avGrid = np.meshgrid(np.linspace(-np.pi,np.pi,num=101), # pole angle
                                                   np.linspace(-8.,8.,num=101),   # pole angle vel
                                                   indexing="ij")
                        cosGrid = np.array([np.cos(ag) for ag in aGrid])
                        sinGrid = np.array([np.sin(ag) for ag in aGrid])
                        selectedActions = trainer.target_actor.forward(Variable(torch.FloatTensor(np.stack((np.array(cosGrid).reshape(-1),np.array(sinGrid).reshape(-1),avGrid.reshape(-1)),axis=1)),volatile=True)).data.numpy()
                        selectedActions_Qvals = []
                        for nei in range(args.N_e):
                                selectedActions_Qvals += [trainer.target_critics[nei].forward(Variable(torch.FloatTensor(np.stack((np.array(cosGrid).reshape(-1),np.array(sinGrid).reshape(-1),avGrid.reshape(-1)),axis=1)),volatile=True),
                                                                                              Variable(torch.FloatTensor(selectedActions),volatile=True)).data.numpy()]
                        axAct.clear();axAngVel.clear();axAng.clear();axR.clear();axActionSpace.clear()
                        axAct.plot([eah[0][0] for eah in epActionHist],"xk",markersize=0.75)
                        axAng.plot([angle_normalize(esh[0][0]) for esh in epStateHist],"-r")
                        axAngVel.plot([esh[1][0] for esh in epStateHist],"-b")
                        axR.plot([erh[0] for erh in epRhist],"-k")
                        axActionSpace.contourf(aGrid,avGrid,
                                               selectedActions.reshape(*aGrid.shape),cmap=plt.cm.seismic,
                                               vmin=-2.0,vmax=2.0)
                        for nei in range(args.N_e):
                                Qgrid[nei].clear()
                                Qgrid[nei].contourf(aGrid,avGrid,
                                                    selectedActions_Qvals[nei].reshape(*aGrid.shape),cmap=plt.cm.rainbow)
                        axQmean.clear()
                        axQmean.contourf(aGrid,avGrid,
                                          np.mean(np.stack(selectedActions_Qvals,axis=1),axis=1).reshape(*aGrid.shape),cmap=plt.cm.rainbow)
                #         axHullAng.set_xticklabels([]); axHullVel.set_xticklabels([]); axJointAngle.set_xticklabels([]); axJointSpeed1.set_xticklabels([]); axJointSpeed2.set_xticklabels([]); axAction1.set_xticklabels([]); axAction2.set_xticklabels([]);
                #         axR.set_xlabel("time")
                #         axHullAng.set_ylabel("Hull ang"); axHullVel.set_ylabel("Hull ang vel"); axJointAngle.set_ylabel("joint"); axJointSpeed1.set_ylabel("leg 1 speeds"); axJointSpeed2.set_ylabel("leg 2 speeds"); axAction1.set_ylabel("leg 1 actions"); axAction2.set_ylabel("leg 2 actions"); axR.set_ylabel("R");
                #         axJointAngle.set_ylim((-1.2,1.2))
                        fig_eval.canvas.draw()
                        plt.pause(0.2)
        all_reward_hist += [(_ep,episodeR)]
        print 'EPISODE :- ', _ep, t, evalEpisode_p, episodeR
                
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
