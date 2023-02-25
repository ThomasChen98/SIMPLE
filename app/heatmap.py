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

from utils.files import load_selected_models, write_results
from utils.register import get_environment
from utils.agents import Agent

import config


def main(args):
    # setup logger
    logger.configure(config.LOGDIR)

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
        checkpoint = loaded['checkpoint']
        heatmap_plot(total_rewards_normalized, checkpoint, args)
        return
    
    # make environment with seed
    env = get_environment(args.env_name)(verbose = args.verbose, manual = args.manual)
    env.seed(args.seed)
    set_global_seeds(args.seed)

    # load the policies
    checkpoint = np.arange(args.arange[0],args.arange[1],args.arange[2])
    logger.info(f'\nLoading {args.env_name} models...')
    models, model_list = load_selected_models(env,checkpoint)
    policy_num = len(model_list)
    
    # total reward
    total_rewards = np.zeros((policy_num, policy_num, env.n_players))

    # agents
    agents = []
    
    # play games
    logger.info(f'\nPlaying {args.games} games...')
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
    save_name = f'./heatmap/{args.env_name}_{args.arange[0]}_{args.arange[1]}_{args.arange[2]}_g{args.games}'
    np.savez_compressed(save_name, total_rewards_normalized=total_rewards_normalized, checkpoint=checkpoint)

    # plot
    heatmap_plot(total_rewards_normalized, checkpoint, args)
 

def heatmap_plot(total_rewards_normalized, checkpoint, args):
    # convert to dataframe for plotting
    heat_data_P1 = total_rewards_normalized[:,:,0]
    heat_data_P2 = total_rewards_normalized[:,:,1]
    heatmap_ticks = ["model_"+str(x) for x in checkpoint]
    df_P1 = pd.DataFrame(data=heat_data_P1, index=heatmap_ticks, columns=heatmap_ticks)
    df_P2 = pd.DataFrame(data=heat_data_P2, index=heatmap_ticks, columns=heatmap_ticks)

    # generate heat plot
    sns.set(rc={'figure.figsize':(15,12)})
    ax = sns.heatmap(df_P1, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="vlag")
    ax.set_title(f"{args.env_name} player 1 average score with {args.games} gameplays".title(),fontsize=25)
    ax.set_xlabel("Player 2", fontsize=20)
    ax.set_ylabel("Player 1", fontsize=20)
    ax.xaxis.tick_top()
    plt.xticks(rotation=45)
    fig = ax.get_figure()
    fig.savefig(f'./heatmap/P1_{args.env_name}_{args.arange[0]}_{args.arange[1]}_{args.arange[2]}_g{args.games}.png') 

    fig.clf()
    ax = sns.heatmap(df_P2, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="vlag")
    ax.set_title(f"{args.env_name} player 2 average score with {args.games} gameplays".title(),fontsize=25)
    ax.set_xlabel("Player 2", fontsize=20)
    ax.set_ylabel("Player 1", fontsize=20)
    ax.xaxis.tick_top()
    plt.xticks(rotation=45)
    fig = ax.get_figure()
    fig.savefig(f'./heatmap/P2_{args.env_name}_{args.arange[0]}_{args.arange[1]}_{args.arange[2]}_g{args.games}.png')


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
  parser.add_argument("--debug", "-d",  action = 'store_true', default = False
                , help="Show logs to debug level")
  parser.add_argument("--env_name", "-e",  type = str, default = 'TicTacToe'
                , help="Which game to play?")
  parser.add_argument("--games", "-g", type = int, default = 1
                , help="Number of games to play)")
  parser.add_argument("--load", "-l",  type = str, default = None
                , help="Which npz to load for plotting?")
  parser.add_argument("--manual", "-m",  action = 'store_true', default = False
                , help="Manual update of the game state on step")
  parser.add_argument("--n_players", "-n", type = int, default = 3
                , help="Number of players in the game (if applicable)")
  parser.add_argument("--randomise_players", "-r",  action = 'store_true', default = False
                , help="Randomise the player order")
  parser.add_argument("--seed", "-s",  type = int, default = 17
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