### Generate heatmap between agents trained by different methods
### Author: Yuxin Chen
### Date: Mar 7, 2023

### Sample usage
# sudo docker-compose exec app python3 tournamentCross.py -e tictactoe -g 100 -ld data
# sudo docker-compose exec app python3 tournamentCross.py -e tictactoe -g 100 -l tictactoe_g100.npz

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

from utils.files import load_all_best_models
from utils.register import get_environment
from utils.agents import Agent

import config


def main(args):
    start_time = MPI.Wtime()

    # setup logger
    logger.configure(config.TOURNAMENTLOGDIR)

    if args.debug:
        logger.set_level(config.DEBUG)
    else:
        logger.set_level(config.INFO)
    
    # if load previous data, directly plot the heatmap
    if args.load != None and not os.path.exists(os.path.join(config.HEATMAPDIR, args.load)):
        raise Exception(f'{args.load} does not exist!')
    elif args.load != None:
        logger.info(f'\nLoading {args.load} data and plot heatmap...')
        loaded = np.load(os.path.join(config.HEATMAPDIR, args.load))
        total_rewards_normalized = loaded['total_rewards_normalized']
        model_list = loaded['model_list']
        heatmap_plot(total_rewards_normalized, model_list, args)
        return
    
    # make environment with seed
    env = get_environment(args.env_name)(verbose = args.verbose, manual = args.manual)
    env.seed(args.seed)
    set_global_seeds(args.seed)

    # load the policies
    policy_dir = ['SP_connect4_best_15M_s5', 'PP_connect4_best_15M_s5']
    models, model_list = load_all_best_models(args.load_dir, policy_dir, env)
    policy_num = len(models)

    # total reward
    total_rewards = np.zeros((policy_num, policy_num, env.n_players))

    # agents
    agents = []
    
    # play games
    logger.info(f'\nPlaying {args.games} games for each of {policy_num} policies...')
    for i in range(policy_num):
        for j in range(policy_num):
            # set up pairing agents
            agents.append(Agent('P1', models[i]))
            agents.append(Agent('P2', models[j]))
            logger.debug(f'Pair {i+1}-{j+1}: P1 = {model_list[i]}: {agents[0]}, P2 = {model_list[j]}: {agents[1]}')

            for game in range(args.games):
                # reset env
                obs = env.reset()
                done = False
                logger.debug(f'Gameplay {i+1}-{j+1} #{game+1} start')

                # shuffle player order
                players = agents[:]
                logger.debug(f'Gameplay {i+1}-{j+1} #{game+1} P1 = {players[0]}, P2 = {players[1]}')
                if args.randomise_players:
                    random.shuffle(players)

                # debug info
                for index, player in enumerate(players):
                    logger.debug(f'Gameplay {i+1}-{j+1} #{game+1}: Player {index+1} = {player.name}')

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
                        total_rewards[i][j][roster[player.name]] += r

                    # pause after each turn to wait for user to continue
                    if args.cont:
                        input('Press any key to continue')
                    
                    env.render()
                    
                    logger.debug(f"Gameplay {i+1}-{j+1} #{game+1} step: {total_rewards[i][j]}")
            
                logger.debug(f"Gameplay {i+1}-{j+1} #{game+1} finished: {total_rewards[i][j]}")
            
            logger.info(f"Gameplay {i+1}-{j+1} finished")

            # reset agents
            agents = []

    env.close()

    # normalize total reward
    total_rewards_normalized = total_rewards / args.games

    # save data
    save_name = f'./heatmap/{args.env_name}_g{args.games}'
    np.savez_compressed(save_name, total_rewards_normalized=total_rewards_normalized, model_list=model_list)

    # plot
    heatmap_plot(total_rewards_normalized, model_list, args)
    logger.info(f"\nGenerate tournament heatmap")
    
    # calculate processing time
    end_time = MPI.Wtime()
    logger.info(f"\nProcessing time: {end_time-start_time}")


def heatmap_plot(total_rewards_normalized, model_list, args):
    # convert to dataframe for plotting
    heat_data_P1 = total_rewards_normalized[:,:,0]
    heat_data_P2 = total_rewards_normalized[:,:,1]
    P1_ticks = [str(x) for x in model_list]
    P2_ticks = [str(x) for x in model_list]
    df_P1 = pd.DataFrame(data=heat_data_P1, index=P1_ticks, columns=P1_ticks)
    df_P2 = pd.DataFrame(data=heat_data_P2, index=P2_ticks, columns=P2_ticks)

    # set title & labels
    P1_title = f"{args.env_name} player 1 average score with {args.games} gameplays"
    P2_title = f"{args.env_name} player 2 average score with {args.games} gameplays"
    xlabel = f"Player 2"
    ylabel = f"Player 1"
    P1_savename = f'./heatmap/{args.env_name}_P1_g{args.games}.png'
    P2_savename = f'./heatmap/{args.env_name}_P2_g{args.games}.png'

    # generate heat plot
    sns.set(rc={'figure.figsize':(15,13)})
    ax = sns.heatmap(df_P1, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap=args.cmap)
    ax.set_title(P1_title.title(),fontsize=25)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.xaxis.tick_top()
    plt.xticks(rotation=45)
    fig = ax.get_figure()
    fig.savefig(P1_savename) 
    fig.clf()

    ax = sns.heatmap(df_P2, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap=args.cmap)
    ax.set_title(P2_title.title(),fontsize=25)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.xaxis.tick_top()
    plt.xticks(rotation=45)
    fig = ax.get_figure()
    fig.savefig(P2_savename)
    fig.clf()


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
  parser.add_argument("--cmap", "-cm", type = str, default = "Spectral"
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