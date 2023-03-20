

import os
import sys
import random
import csv
import time
import numpy as np

from mpi4py import MPI

from shutil import rmtree
from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy

from utils.register import get_network_arch

import config

from stable_baselines import logger


def write_results(players, game, games, episode_length):
    
    out = {'game': game
    , 'games': games
    , 'episode_length': episode_length
    , 'p1': players[0].name
    , 'p2': players[1].name
    , 'p1_points': players[0].points
    , 'p2_points': np.sum([x.points for x in players[1:]])
    }

    if not os.path.exists(config.RESULTSPATH):
        with open(config.RESULTSPATH,'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=out.keys())
            writer.writeheader()

    with open(config.RESULTSPATH,'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out.keys())
        writer.writerow(out)


def load_model(env, name):

    filename = os.path.join(config.MODELDIR, env.name, f'{MPI.COMM_WORLD.Get_rank()}', name)
    if os.path.exists(filename):
        logger.info(f'Loading {name}')
        cont = True
        while cont:
            try:
                ppo_model = PPO1.load(filename, env=env)
                cont = False
            except Exception as e:
                time.sleep(5)
                print(e)
    
    elif name == 'base.zip':
        cont = True
        while cont:
            try:
                
                rank = MPI.COMM_WORLD.Get_rank()

                ppo_model = PPO1(get_network_arch(env.name), env=env)
                ppo_model.save(os.path.join(config.MODELDIR, env.name, f'{rank}', 'base.zip'))
                logger.info(f'Saving base.zip PPO model...')

                cont = False
            except IOError as e:
                sys.exit(f'Check zoo/{env.name}/ exists and read/write permission granted to user')
            except Exception as e:
                logger.error(e)
                time.sleep(2)
                
    else:
        raise Exception(f'\n{filename} not found')
    
    return ppo_model

def load_model_with_rank(env, name, threadID, opp_rank):

    my_rank = MPI.COMM_WORLD.Get_rank()

    filename = os.path.join(config.MODELDIR, env.name, f'thread_{threadID}', f'rank_{opp_rank}', name)
    if os.path.exists(filename):
        logger.info(f'+++Thread {threadID} Rank {my_rank}+++ Rank {my_rank} loading {name}')
        cont = True
        while cont:
            try:
                ppo_model = PPO1.load(filename, env=env)
                cont = False
            except Exception as e:
                time.sleep(5)
                print(e)
    
    elif name == 'base.zip':
        cont = True
        while cont:
            try:
                ppo_model = PPO1(get_network_arch(env.name), env=env)
                ppo_model.save(os.path.join(config.MODELDIR, env.name, f'thread_{threadID}', f'rank_{opp_rank}', f'base_{threadID}_{opp_rank}.zip'))
                logger.info(f'+++Thread {threadID} Rank {my_rank}+++ Rank {my_rank} saving base_{threadID}_{opp_rank}.zip PPO model...')

                cont = False
            except IOError as e:
                sys.exit(f'Check zoo/{env.name}/ exists and read/write permission granted to user')
            except Exception as e:
                logger.error(e)
                time.sleep(2)
                
    else:
        raise Exception(f'\n{filename} not found')
    
    return ppo_model

def load_model_with_rank_in_pool(env, name, opp_rank):

    my_rank = MPI.COMM_WORLD.Get_rank()

    filename = os.path.join(config.POOLDIR, env.name, f'{opp_rank}', name)
    if os.path.exists(filename):
        logger.info(f'Rank {my_rank} loading {name}')
        cont = True
        while cont:
            try:
                ppo_model = PPO1.load(filename, env=env)
                cont = False
            except Exception as e:
                time.sleep(5)
                print(e)
                
    else:
        raise Exception(f'\n{filename} not found')
    
    return ppo_model

def load_all_models(env, threadID):
    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env.name, f'thread_{threadID}', f'rank_{MPI.COMM_WORLD.Get_rank()}')) if f.startswith("_model")]
    modellist.sort()
    models = [load_model_with_rank(env, 'base.zip', threadID, MPI.COMM_WORLD.Get_rank())]
    for model_name in modellist:
        models.append(load_model_with_rank(env, model_name, threadID, MPI.COMM_WORLD.Get_rank()))
    return models

def load_selected_models(dir, env, rank, checkpoints):
    modellist = []
    for cp in checkpoints:
        for f in os.listdir(os.path.join(dir, f'{rank}')):
            if f.startswith("_model_"+str(rank).zfill(2)+"_"+str(cp).zfill(5)):
                modellist.append(f)
    modellist.sort()
    models = []
    for model_name in modellist:
        models.append(load_model_custom(os.path.join(dir, f'{rank}'), env=env, name = model_name))
    return models, modellist

def load_model_custom(dir, env, name):

    filename = os.path.join(dir, name)
    if os.path.exists(filename):
        logger.info(f'Loading {name}')
        cont = True
        while cont:
            try:
                ppo_model = PPO1.load(filename, env=env)
                cont = False
            except Exception as e:
                time.sleep(5)
                print(e)
    else:
        raise Exception(f'\n{filename} not found')
    
    return ppo_model

def load_all_best_models(dir, policy_list, env):
    models = []
    modellist = []
    load_path = os.path.join(dir, policy_list, 'models')
    loadlist = [d for d in os.listdir(load_path)]
    loadlist.sort()
    for d in loadlist:
        for f in os.listdir(os.path.join(load_path, d)):
            if f.startswith("best_model"):
                filename = os.path.join(load_path, d, f)
                longnamelist = filename.split('/')
                longname = longnamelist[1]
                keywordlist = longname.split('_')
                modelname = keywordlist[0]+'_'+d
                logger.info(f'Load {filename} as {modelname}')
                # load model
                cont = True
                while cont:
                    try:
                        ppo_model = PPO1.load(filename, env=env)
                        cont = False
                    except Exception as e:
                        time.sleep(5)
                        print(e)
                # append model
                models.append(ppo_model)
                modellist.append(modelname)
    return models, modellist

def get_best_model_name(threadID, env_name):
    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env_name, f'thread_{threadID}', f'rank_{MPI.COMM_WORLD.Get_rank()}')) if f.startswith("_model")]
    
    if len(modellist)==0:
        filename = None
    else:
        modellist.sort()
        filename = modellist[-1]
        
    return filename

def get_opponent_best_model_name(env_name, opp_rank):
    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env_name, f'{opp_rank}')) if f.startswith("_model")]
    
    if len(modellist)==0:
        modellist_with_base = [f for f in os.listdir(os.path.join(config.MODELDIR, env_name, f'{opp_rank}'))]
        if len(modellist_with_base) != 0:
            filename = f'base_{opp_rank}.zip'
        else:
            filename = None
    else:
        modellist.sort()
        filename = modellist[-1]
        
    return filename

def get_random_model_name_with_rank(env_name, opp_rank):
    modellist = [f for f in os.listdir(os.path.join(config.POOLDIR, env_name, f'{opp_rank}')) if f.startswith("_model")]
    modellist_with_base = [f for f in os.listdir(os.path.join(config.POOLDIR, env_name, f'{opp_rank}'))]
    
    if len(modellist)==0: # no saved model yet
        if len(modellist_with_base) != 0:
            filename = f'base_{opp_rank}.zip'
        else:
            filename = None
    elif len(modellist)==1: # only one saved model, use base model
        filename = f'base_{opp_rank}.zip'
    else:
        modellist.sort()
        filename = random.choice(modellist)
        
    return filename

def get_model_stats(filename):
    if filename is None:
        generation = 0
        timesteps = 0
        best_rules_based = -np.inf
        best_reward = -np.inf
    else:
        stats = filename.split('_')
        generation = int(stats[4])
        best_rules_based = float(stats[5])
        best_reward = float(stats[6])
        timesteps = int(stats[7])
    return generation, timesteps, best_rules_based, best_reward


def reset_logs(threadID):
    try:
        filelist = [ f for f in os.listdir(config.LOGDIR) if f not in ['.gitignore']]
        for f in filelist:
            if os.path.isfile(f):  
                os.remove(os.path.join(config.LOGDIR, f))
            else:
                rmtree(os.path.join(config.LOGDIR, f))
        
        os.makedirs(os.path.join(config.LOGDIR, f'thread_{threadID}'))
        open(os.path.join(config.LOGDIR, f'thread_{threadID}', 'log.txt'), 'a').close()
    
        
    except Exception as e :
        print(e)
        print('Reset logs failed')

def reset_models(model_dir):
    try:
        filelist = [f for f in os.listdir(model_dir) if f not in ['.gitignore']]
        for f in filelist:
            os.remove(os.path.join(model_dir , f))
    except Exception as e :
        print(e)
        print('Reset models failed')