

import os
import sys
import random
import csv
import time
import numpy as np
import time
import re

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

def load_model_with_id(env, name, opp_id):
    filename = os.path.join(config.MODELDIR, env.name, f'thread_{opp_id}', name)
    if os.path.exists(filename):
        logger.info(f'+++Thread {opp_id}+++ Loading {name}')
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
                opp_id_str = str(opp_id).zfill(2)
                ppo_model.save(os.path.join(config.MODELDIR, env.name, f'thread_{opp_id}', f'_base_{opp_id_str}.zip'))
                logger.info(f'+++Thread {opp_id}+++ Saving _base_{opp_id_str}.zip PPO model...')

                cont = False
            except IOError as e:
                sys.exit(f'Check zoo/{env.name}/ exists and read/write permission granted to user')
            except Exception as e:
                logger.error(e)
                time.sleep(2)
                
    else:
        raise Exception(f'\n{filename} not found')
    
    return ppo_model

def load_SP_model_with_id(env, name, opp_id):

    filename = os.path.join(config.POOLDIR, env.name, 'models', f'thread_{opp_id}', name)
    if os.path.exists(filename):
        logger.info(f'+++Self-Play Thread {opp_id}+++ Loading {name}')
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
    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env.name, f'thread_{threadID}')) if f.startswith("_model")]
    modellist.sort()
    models = [load_model_with_id(env, 'base.zip', threadID)]
    for model_name in modellist:
        models.append(load_model_with_id(env, model_name, threadID))
    return models

def load_selected_models(dir, env, rank, checkpoints):
    modellist = []
    for cp in checkpoints:
        for f in os.listdir(os.path.join(dir, f'thread_{rank}')):
            if f.startswith("_model_"+str(rank).zfill(2)+"_"+str(cp).zfill(5)):
                modellist.append(f)
    modellist.sort()
    models = []
    for model_name in modellist:
        models.append(load_model_with_name(os.path.join(dir, f'thread_{rank}'), env=env, name = model_name))
    return models, modellist

def load_model_with_name(dir, env, name):

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

def get_best_model_name(env_name, threadID):
    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env_name, f'thread_{threadID}')) if f.startswith("_model")]
    
    if len(modellist)==0:
        modellist_with_base = [f for f in os.listdir(os.path.join(config.MODELDIR, env_name, f'thread_{threadID}'))]
        if len(modellist_with_base) != 0:
            opp_id_str = str(threadID).zfill(2)
            filename = f'_base_{opp_id_str}.zip'
        else:
            filename = None
    else:
        modellist.sort()
        filename = modellist[-1]
        
    return filename

def get_random_SP_model_name(env_name, threadID):
    modellist = [f for f in os.listdir(os.path.join(config.POOLDIR, env_name, 'models', f'thread_{threadID}')) if f.startswith("_")]
    
    modellist.sort()
    filename = random.choice(modellist)
        
    return filename

def get_model_length(env_name, threadID):
    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env_name, f'thread_{threadID}')) if f.startswith("_model")]
    
    return len(modellist)

def get_opponent_length(env_name, threadID):
    with open(os.path.join(config.MODELDIR, env_name, f'thread_{threadID}', 'opponents.txt')) as f:
        modellist = f.readlines()
    
    return len(modellist)

def get_current_opponent_name_id(env_name, threadID):
    with open(os.path.join(config.MODELDIR, env_name, f'thread_{threadID}', 'opponents.txt')) as f:
        modellist = f.readlines()
    if len(modellist) == 0:
        return -1
    else:
        modellist.sort()
        filename = modellist[-1]
        stats = re.split(r'[._]',filename)
        id = int(stats[2])
        name = filename[5:]
    
    return name, id

def get_model_stats(filename):
    stats = filename.split('_')
    if filename is None or len(stats) < 7:
        generation = 0
        timesteps = 0
        best_rules_based = -np.inf
        best_reward = -np.inf
    else:
        generation = int(stats[3])
        best_rules_based = float(stats[4])
        best_reward = float(stats[5])
        timesteps = int(stats[6])
    return generation, timesteps, best_rules_based, best_reward


def reset_logs(threadID):
    if os.path.exists(os.path.join(config.LOGDIR, f'thread_{threadID}')):
        rmtree(os.path.join(config.LOGDIR, f'thread_{threadID}'))
    os.makedirs(os.path.join(config.LOGDIR, f'thread_{threadID}'))
    open(os.path.join(config.LOGDIR, f'thread_{threadID}', 'log.txt'), 'a').close()

def reset_models(model_dir):
    try:
        filelist = [f for f in os.listdir(model_dir) if f not in ['.gitignore']]
        for f in filelist:
            os.remove(os.path.join(model_dir , f))
    except Exception as e :
        print(e)
        print('Reset models failed')

def reset_models_PP(model_dir, env_name, threadID):
    try:
        rmtree(os.path.join(config.MODELDIR, env_name))
    except Exception as e :
        pass
    time.sleep(5)
    os.makedirs(model_dir)
    with open(os.path.join(model_dir, 'opponents.txt'), "a") as f:
        opp_id_str = str(threadID).zfill(2)
        filename = f'00000_base_{opp_id_str}.zip'
        f.write(filename)