### Generate heatmap between agents trained by different methods
### Author: Yuxin Chen
### Date: Mar 7, 2023

### Sample usage
# sudo docker-compose exec app mpirun -np 25 python3 tournamentCrossPolicy.py -e tictactoe -r -g 100 -ld data
# sudo docker-compose exec app python3 tournamentCrossPolicy.py -e tictactoe -g 100 -l TTT/tictactoe_std_g100.npz

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
    # if load previous data, directly plot the heatmap
    if args.load != None and not os.path.exists(os.path.join(config.TOURNAMENTDIR, args.load)):
        raise Exception(f'{args.load} does not exist!')
    elif args.load != None:
        logger.info(f'\nLoading {args.load} data and plot heatmap...')
        loaded = np.load(os.path.join(config.TOURNAMENTDIR, args.load))
        # world_total_rewards_stats = loaded['world_total_rewards_stats']
        # policy_dir = loaded['policy_dir']
        # heatmap_plot_stats(world_total_rewards_stats, policy_dir, args, opt='std')
        total_rewards_normalized = loaded['total_rewards_normalized']
        ego_model_list = loaded['ego_model_list']
        opp_model_list = loaded['opp_model_list']
        heatmap_plot(total_rewards_normalized, ego_model_list, opp_model_list, ['PP_s5','PP_s5'], args)
        return
    
    start_time = MPI.Wtime()

    rank = MPI.COMM_WORLD.Get_rank()

    # setup logger
    logger.configure(config.TOURNAMENTLOGDIR)

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
    policy_dir = ['SP_C4_10M_s8', 'PP_C4_10M_s3', 'PP_C4_10M_s5',\
                  'FCP_C4_20M_p3', 'FCP_C4_20M_p5']
    
    # check mpi rank
    if MPI.COMM_WORLD.Get_size() != len(policy_dir)**2:
        raise Exception(f'MPI processors number should be {len(policy_dir)**2}!')
    
    # load models for this rank
    ego_policy_dir = rank//len(policy_dir)
    opp_policy_dir = rank%len(policy_dir)
    ego_models, ego_model_list = load_all_best_models(args.load_dir, policy_dir[ego_policy_dir], env)
    opp_models, opp_model_list = load_all_best_models(args.load_dir, policy_dir[opp_policy_dir], env)
    ego_policy_num = len(ego_models)
    opp_policy_num = len(opp_models)

    # total reward
    total_rewards = np.zeros((ego_policy_num, opp_policy_num, env.n_players))

    # agents
    agents = []
    
    # play games
    logger.info(f'\nPlaying {args.games} games for each of {ego_policy_num}x{opp_policy_num} policies...')
    for i in range(ego_policy_num):
        for j in range(opp_policy_num):
            # set up pairing agents
            agents.append(Agent('P1', ego_models[i]))
            agents.append(Agent('P2', opp_models[j]))
            logger.debug(f'Pair {i}-{j}: P1 = {ego_model_list[i]}: {agents[0]}, P2 = {opp_model_list[j]}: {agents[1]}')

            for game in range(args.games):
                # reset env
                obs = env.reset()
                done = False
                logger.debug(f'Gameplay {i}-{j} #{game} start')

                # shuffle player order
                players = agents[:]
                logger.debug(f'Gameplay {i}-{j} #{game} P1 = {players[0]}, P2 = {players[1]}')
                if args.randomise_players:
                    random.shuffle(players)

                # debug info
                for index, player in enumerate(players):
                    logger.debug(f'Gameplay {i}-{j} #{game}: Player {index} = {player.name}')

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
                    
                    logger.debug(f"Gameplay {i}-{j} #{game} step: {total_rewards[i][j]}")
            
                logger.debug(f"Gameplay {i}-{j} #{game} finished: {total_rewards[i][j]}")
            
            logger.info(f"Gameplay {i}-{j} finished")

            # reset agents
            agents = []

    env.close()

    # normalize total reward
    total_rewards_normalized = total_rewards / args.games

    # save data
    ego_name_vec = policy_dir[ego_policy_dir].split('_')
    opp_name_vec = policy_dir[opp_policy_dir].split('_')
    ego_name = f'{ego_name_vec[0].upper()}_{ego_name_vec[-1]}'
    opp_name = f'{opp_name_vec[0].upper()}_{opp_name_vec[-1]}'
    save_name = f'./plot_tournament/{args.env_name}_{ego_name}vs{opp_name}_g{args.games}'
    np.savez_compressed(save_name, total_rewards_normalized=total_rewards_normalized,\
                        ego_model_list=ego_model_list, opp_model_list=opp_model_list)

    # plot
    heatmap_plot(total_rewards_normalized, ego_model_list, opp_model_list, [ego_name, opp_name], args)
    logger.info(f"\nGenerate tournament heatmap")

    # plot total map
    if rank == 0:
        world_total_rewards_avg = np.zeros((len(policy_dir), len(policy_dir), env.n_players))
        world_total_rewards_std = np.zeros((len(policy_dir), len(policy_dir), env.n_players))
        world_total_rewards_avg[0,0,:] = total_rewards_normalized.mean(axis=(0, 1))
        world_total_rewards_std[0,0,:] = total_rewards_normalized.std(axis=(0, 1))
        for i in range( 1, MPI.COMM_WORLD.Get_size() ):
            current_total_rewards_avg = np.zeros(env.n_players)
            current_total_rewards_std = np.zeros(env.n_players)
            MPI.COMM_WORLD.Recv( [current_total_rewards_avg, MPI.DOUBLE], source=i, tag=i )
            logger.info(f"{i}th normalized total_reward_avg received")
            MPI.COMM_WORLD.Recv( [current_total_rewards_std, MPI.DOUBLE], source=i, tag=i+100 )
            logger.info(f"{i}th normalized total_reward_std received")
            col = i//len(policy_dir)
            row = i%len(policy_dir)
            world_total_rewards_avg[col,row,:] = current_total_rewards_avg
            world_total_rewards_std[col,row,:] = current_total_rewards_std

        # save data
        total_name = f'./plot_tournament/{args.env_name}_avg_g{args.games}'
        np.savez_compressed(total_name, world_total_rewards_stats=world_total_rewards_avg, policy_dir=policy_dir)
        total_name = f'./plot_tournament/{args.env_name}_std_g{args.games}'
        np.savez_compressed(total_name, world_total_rewards_stats=world_total_rewards_std, policy_dir=policy_dir)

        # plot
        heatmap_plot_stats(world_total_rewards_avg, policy_dir, args, opt='avg')
        logger.info(f"\nGenerate average tournament heatmap")
        heatmap_plot_stats(world_total_rewards_std, policy_dir, args, opt='std')
        logger.info(f"\nGenerate tournament heatmap std")
    else:
        MPI.COMM_WORLD.Send( [total_rewards_normalized.mean(axis=(0, 1)), MPI.DOUBLE], dest=0, tag=rank )
        logger.info(f"\nRank {rank} normalized total_reward_avg sent")
        MPI.COMM_WORLD.Send( [total_rewards_normalized.std(axis=(0, 1)), MPI.DOUBLE], dest=0, tag=rank+100 )
        logger.info(f"\nRank {rank} normalized total_reward_std sent")
    
    # calculate processing time
    end_time = MPI.Wtime()
    logger.info(f"\nProcessing time: {end_time-start_time}")


def heatmap_plot(total_rewards_normalized, ego_model_list, opp_model_list, name_list, args):
    # convert to dataframe for plotting
    heat_data_P1 = total_rewards_normalized[:,:,0]
    heat_data_P2 = total_rewards_normalized[:,:,1]
    P1_ticks = [str(x) for x in ego_model_list]
    P2_ticks = [str(x) for x in opp_model_list]
    df_P1 = pd.DataFrame(data=heat_data_P1, index=P1_ticks, columns=P2_ticks)
    df_P2 = pd.DataFrame(data=heat_data_P2, index=P1_ticks, columns=P2_ticks)

    # set title & labels
    P1_title = f"{args.env_name} {name_list[0]} vs. {name_list[1]} row player average score with {args.games} gameplays"
    P2_title = f"{args.env_name} {name_list[0]} vs. {name_list[1]} column player average score with {args.games} gameplays"
    xlabel = f"Random seeds for column player"
    ylabel = f"Random seeds for row player"
    P1_savename = f'./plot_tournament/{args.env_name}_{name_list[0]}vs{name_list[1]}_P1_g{args.games}.pdf'
    P2_savename = f'./plot_tournament/{args.env_name}_{name_list[0]}vs{name_list[1]}_P2_g{args.games}.pdf'

    # generate heat plot
    sns.set(rc={'figure.figsize':(15,13)})
    ax = sns.heatmap(df_P1, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap=args.cmap, square = True, cbar_kws={"shrink": 1})
    ax.set_title(P1_title.title(), fontsize=25, y=1.05)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    # ax.xaxis.tick_top()
    plt.subplots_adjust(right=1)
    plt.xticks(rotation=45)
    fig = ax.get_figure()
    fig.savefig(P1_savename) 
    fig.clf()

    ax = sns.heatmap(df_P2, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap=args.cmap, square = True, cbar_kws={"shrink": 1})
    ax.set_title(P2_title.title(), fontsize=25, y=1.05)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    # ax.xaxis.tick_top()
    plt.subplots_adjust(right=1)
    plt.xticks(rotation=45)
    fig = ax.get_figure()
    fig.savefig(P2_savename)
    fig.clf()

def heatmap_plot_stats(world_total_rewards_stats, policy_dir, args, opt='avg'):
    # convert to short name
    policy_name = []
    for f in policy_dir:
        name_vec = f.split('_')
        policy_name.append(f'{name_vec[0]}_{name_vec[-1]}')
    # convert to dataframe for plotting
    heat_data_P1 = world_total_rewards_stats[:,:,0]
    heat_data_P2 = world_total_rewards_stats[:,:,1]
    P1_ticks = [str(x) for x in policy_name]
    P2_ticks = [str(x) for x in policy_name]
    df_P1 = pd.DataFrame(data=heat_data_P1, index=P1_ticks, columns=P2_ticks)
    df_P2 = pd.DataFrame(data=heat_data_P2, index=P1_ticks, columns=P2_ticks)

    # set title & labels
    P1_title = f"{args.env_name} row player average score with {args.games} gameplays"
    P2_title = f"{args.env_name} column player average score with {args.games} gameplays"
    xlabel = f"Training methods for column player"
    ylabel = f"Training methods for row player"
    P1_savename = f'./plot_tournament/{args.env_name}_avg_P1_g{args.games}.pdf'
    P2_savename = f'./plot_tournament/{args.env_name}_avg_P2_g{args.games}.pdf'
    if opt == 'std':
        P1_title = f"{args.env_name} row player score std with {args.games} gameplays"
        P2_title = f"{args.env_name} column player score std with {args.games} gameplays"
        xlabel = f"Training methods for column player"
        ylabel = f"Training methods for row player"
        P1_savename = f'./plot_tournament/{args.env_name}_std_P1_g{args.games}.pdf'
        P2_savename = f'./plot_tournament/{args.env_name}_std_P2_g{args.games}.pdf'

    # generate heat plot
    sns.set(rc={'figure.figsize':(15, 13)})
    ax = sns.heatmap(df_P1, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap=args.cmap, square = True, cbar_kws={"shrink": 1})
    ax.set_title(P1_title.title(), fontsize=25, y=1.05)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    # ax.xaxis.tick_top()
    plt.subplots_adjust(right=1)
    plt.xticks(rotation=45)
    fig = ax.get_figure()
    fig.savefig(P1_savename) 
    fig.clf()

    ax = sns.heatmap(df_P2, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap=args.cmap, square = True, cbar_kws={"shrink": 1})
    ax.set_title(P2_title.title(), fontsize=25, y=1.05)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    # ax.xaxis.tick_top()
    plt.subplots_adjust(right=1)
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