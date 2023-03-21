### Generate Elo score
### Author: Yuxin Chen
### Date: Mar 14, 2023

# sudo docker-compose exec app mpirun -np 5 python3 elo.py -e tictactoe -r -g 100 -a 1 25 2 -p 5 -ld data/SP_tictactoe_10M_s5/models

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from stable_baselines import logger
from stable_baselines.common import set_global_seeds

from mpi4py import MPI

from utils.files import load_selected_models
from utils.register import get_environment
from utils.agents import Agent

import config


def main(args):
    
    start_time = MPI.Wtime()
    # check mpi rank
    rank = MPI.COMM_WORLD.Get_rank()
    if MPI.COMM_WORLD.Get_size() != args.population:
        raise Exception(f'MPI processors number should be {args.population}!')

    # setup logger
    logger.configure(config.ELOLOGDIR)

    if args.debug:
        logger.set_level(config.DEBUG)
    else:
        logger.set_level(config.INFO)
    
    # make environment with seed
    env = get_environment(args.env_name)(verbose = args.verbose, manual = args.manual)
    workerseed = args.seed + 10000 * rank
    env.seed(workerseed)
    set_global_seeds(workerseed)

    # load the policies
    checkpoint = np.arange(args.arange[0],args.arange[1],args.arange[2])
    logger.info(f'\n##### Rank {rank+1} #####\nLoading {args.env_name} {rank+1} models...')
    models, model_list = load_selected_models(args.load_dir,env,rank,checkpoint)
    policy_num = len(models)
    
    # scores and elo rate
    actual_score = np.zeros(policy_num)
    expected_score = np.zeros(policy_num)
    elo = np.zeros(policy_num)

    # agents
    agents = []
    
    # play games
    logger.info(f'\nPlaying {args.games} games for each of {policy_num} policies...')
    for i in range(1, policy_num): # define the first checkpoint's rate as 0
        # set up pairing agents
        agents.append(Agent('P1', models[i]))
        agents.append(Agent('P2', models[i-1]))
        logger.debug(f'Pair {i}-{i-1}: P1 = {model_list[i]}: {agents[0]}, P2 = {model_list[i-1]}: {agents[1]}')

        for game in range(args.games):
            # reset env
            obs = env.reset()
            done = False
            rewards = np.zeros(env.n_players)
            logger.debug(f'Gameplay {i}-{i-1} #{game+1} start')

            # shuffle player order
            players = agents[:]
            logger.debug(f'Gameplay {i}-{i-1} #{game+1} P1 = {players[0]}, P2 = {players[1]}')
            if args.randomise_players:
                random.shuffle(players)

            # debug info
            for index, player in enumerate(players):
                logger.debug(f'Gameplay {i}-{i-1} #{game+1}: Player {index+1} = {player.name}')

            while not done:
                # current player info
                current_player = players[env.current_player_num]
                env.render()
                logger.debug(f'\nCurrent player name: {current_player.name}, id: {current_player.id}')
                
                # current player action
                logger.debug(f'\n{current_player.name} model choices')
                action = current_player.choose_action(env, choose_best_action = args.best, mask_invalid_actions = True)
                
                # step env
                obs, reward, done, _ = env.step(action)
                
                # calculate reward
                roster = {'P1': 0, 'P2': 1}
                for r, player in zip(reward, players):
                    logger.debug(f'Player {player.name} + {r}')
                    rewards[roster[player.name]] += r

                # pause after each turn to wait for user to continue
                if args.cont:
                    input('Press any key to continue')
                
                env.render()
                
                logger.debug(f"Gameplay {i}-{i-1} #{game+1} step: {rewards}")
            
            # update actual score
            if rewards[0] == -1: # loss
                actual_score[i] += 0
            elif rewards[0] == 0: # draw
                actual_score[i] += 0.5
            else: # defeat
                actual_score[i] += 1
            
            logger.debug(f"Gameplay {i}-{i-1} #{game+1} finished: {rewards}, actual score: {actual_score}")
        
        # update expect score
        expected_score[i] = args.games/(1+10**(elo[i-1]-elo[i]/400))

        # update Elo rate
        K = 32
        elo[i] += K*(actual_score[i]-expected_score[i])

        logger.info(f"Gameplay {i}-{i-1} finished, policy {i} ELO rating: {elo[i]}")

        # reset agents
        agents = []

    env.close()

    # plot elo
    if rank == 0:
        world_elo = np.zeros((MPI.COMM_WORLD.Get_size(), policy_num))
        world_elo[0] = elo
        for i in range( 1, MPI.COMM_WORLD.Get_size() ):
            current_elo = np.zeros(policy_num)
            MPI.COMM_WORLD.Recv( [current_elo, MPI.DOUBLE], source=i, tag=i )
            world_elo[i] = current_elo
            logger.info(f"{i+1}th elo received")

        # convert to pandas
        full_df = pd.DataFrame()
        for i in range(MPI.COMM_WORLD.Get_size()):
            d = {'gen': range(policy_num), 'elo': world_elo[i], 'seed': [f'seed {i+1}']*policy_num}
            df = pd.DataFrame(d)
            full_df = pd.concat([full_df, df], axis=0, ignore_index=True)
        
        save_name = f'./plot_elo/{args.env_name}_g{args.games}'
        np.savez_compressed(save_name, world_elo=world_elo)
    
        # plot
        plot_elo(full_df)
    else:
        MPI.COMM_WORLD.Send( [elo, MPI.DOUBLE], dest=0, tag=rank )
        logger.info(f"\nRank {rank+1} elo sent")
    
    # calculate processing time
    end_time = MPI.Wtime()
    if rank == 0:
        logger.info(f"\nProcessing time: {end_time-start_time}")

def plot_elo(full_df):
    figsize=(10, 6)
    sns.set()
    sns.set_context("paper")

    fig, ax = plt.subplots(figsize=figsize)
    g = sns.lineplot(data=full_df, x='gen', y='elo')
    ax.set(xlabel='Model generation', ylabel='ELO rate')
    plt.savefig(os.path.join(config.ELODIR, 'tictactoe_SP.pdf'), bbox_inches='tight')

def cli() -> None:
  """Handles argument extraction from CLI and passing to main().
  Note that a separate function is used rather than in __name__ == '__main__'
  to allow unit testing of cli().
  """
  # Setup argparse to show defaults on help
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)

  parser.add_argument("--arange","-a", nargs = '+', type=int, default = [1,60,15]
                , help="Arange for checkpoint selecting")
  parser.add_argument("--best", "-b", action = 'store_true', default = False
                , help="Make AI agents choose the best move (rather than sampling)")
  parser.add_argument("--cont", "-c",  action = 'store_true', default = False
                , help="Pause after each turn to wait for user to continue")
  parser.add_argument("--cmap", "-cm", type = str, default = "vlag"
                , help="Colormap")
  parser.add_argument("--debug", "-d",  action = 'store_true', default = False
                , help="Show logs to debug level")
  parser.add_argument("--env_name", "-e",  type = str, default = 'tictactoe'
                , help="Which game to play?")
  parser.add_argument("--games", "-g", type = int, default = 1
                , help="Number of games to play)")
  parser.add_argument("--load", "-l",  type = str, default = None
                , help="Which npz to load for plotting?")
  parser.add_argument("--load_dir", "-ld", type = str, default = None
                , help="Which directory to load models?")
  parser.add_argument("--manual", "-m",  action = 'store_true', default = False
                , help="Manual update of the game state on step")
  parser.add_argument("--n_players", "-n", type = int, default = 3
                , help="Number of players in the game (if applicable)")
  parser.add_argument("--population", "-p", type = int, default = 5
                , help="Pupulation size")
  parser.add_argument("--randomise_players", "-r",  action = 'store_true', default = False
                , help="Randomise the player order")
  parser.add_argument("--seed", "-s",  type = int, default = 5 # is different from all trainning env
                , help="Random seed")
  parser.add_argument("--verbose", "-v",  action = 'store_true', default = False
                , help="Show observation on debug logging")

  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()