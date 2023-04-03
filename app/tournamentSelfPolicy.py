### Generate heatmap for self-play agents
### Author: Yuxin Chen
### Date: Feb 24, 2023

### Sample usage
# sudo docker-compose exec app mpirun -np 64 python3 tournamentSelfPolicy.py -e tictactoe -r -g 100 -a 1 271 18 -p 8 -ld data/SP_TTT_20M_s8/models
# sudo docker-compose exec app python3 tournamentSelfPolicy.py -e tictactoe -g 100 -a 1 25 2 -l SP_tictactoe_10M_s5_1.25.2/tictactoe_avg_1.25.2_g100.npz

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
    # if load previous data, directly plot the heatmap
    if args.load != None and not os.path.exists(os.path.join(config.HEATMAPDIR, args.load)):
        raise Exception(f'{args.load} does not exist!')
    elif args.load != None:
        logger.info(f'\nLoading {args.load} data and plot heatmap...')
        loaded = np.load(os.path.join(config.HEATMAPDIR, args.load))
        total_rewards_normalized = loaded['total_rewards_normalized']
        checkpoint = loaded['checkpoint']
        heatmap_plot(total_rewards_normalized, checkpoint, args, opt='avg')
        return
    
    start_time = MPI.Wtime()
    # check mpi rank
    rank = MPI.COMM_WORLD.Get_rank()
    if MPI.COMM_WORLD.Get_size() != args.population**2:
        raise Exception(f'MPI processors number should be {args.population**2}!')

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
    checkpoint = np.arange(args.arange[0],args.arange[1],args.arange[2])
    ego_rank = rank//args.population
    opp_rank = rank%args.population
    logger.info(f'\n##### Rank {rank} #####\nLoading {args.env_name} seed {ego_rank} model as ego, seed {opp_rank} model as opponent...')
    ego_models, ego_model_list = load_selected_models(args.load_dir,env,ego_rank,checkpoint)
    opp_models, opp_model_list = load_selected_models(args.load_dir,env,opp_rank,checkpoint)
    if len(ego_models) != len(opp_models):
        raise Exception(f'# of ego policies and opponent policies does not match!')
    policy_num = len(ego_models)
    
    # total reward
    total_rewards = np.zeros((policy_num, policy_num, env.n_players))

    # agents
    agents = []
    
    # play games
    logger.info(f'\nPlaying {args.games} games for each of {policy_num} policies...')
    for i in range(policy_num):
        for j in range(policy_num):
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
    save_name = f'./plot_tournament/{args.env_name}_{ego_rank}vs{opp_rank}_{args.arange[0]}.{args.arange[1]}.{args.arange[2]}_g{args.games}'
    np.savez_compressed(save_name, total_rewards_normalized=total_rewards_normalized, checkpoint=checkpoint, ranks=[ego_rank, opp_rank])

    # plot
    heatmap_plot(total_rewards_normalized, checkpoint, args, ranks=[ego_rank, opp_rank])
    logger.info(f"\nGenerate tournament heatmap for seed {ego_rank} vs. seed {opp_rank}")

    # plot average heatmap & deviation
    if rank == 0:
        world_total_rewards_normalized = np.zeros((MPI.COMM_WORLD.Get_size(), policy_num, policy_num, env.n_players))
        world_total_rewards_normalized[0,:,:,:] = total_rewards_normalized
        for i in range( 1, MPI.COMM_WORLD.Get_size() ):
            current_total_rewards_normalized = np.zeros((policy_num, policy_num, env.n_players))
            MPI.COMM_WORLD.Recv( [current_total_rewards_normalized, MPI.DOUBLE], source=i, tag=i )
            world_total_rewards_normalized[i,:,:,:] = current_total_rewards_normalized
            logger.info(f"{i}th normalized total_reward received")
        total_rewards_normalized_avg = np.mean(world_total_rewards_normalized, axis=0)
        total_rewards_normalized_std = np.std(world_total_rewards_normalized, axis=0)

        # save data
        avg_name = f'./plot_tournament/{args.env_name}_avg_{args.arange[0]}.{args.arange[1]}.{args.arange[2]}_g{args.games}'
        std_name = f'./plot_tournament/{args.env_name}_std_{args.arange[0]}.{args.arange[1]}.{args.arange[2]}_g{args.games}'
        np.savez_compressed(avg_name, total_rewards_normalized=total_rewards_normalized_avg, checkpoint=checkpoint)
        np.savez_compressed(std_name, total_rewards_normalized=total_rewards_normalized_std, checkpoint=checkpoint)

        # plot
        heatmap_plot(total_rewards_normalized_avg, checkpoint, args, opt='avg')
        logger.info(f"\nGenerate average tournament heatmap")
        heatmap_plot(total_rewards_normalized_std, checkpoint, args, opt='std')
        logger.info(f"\nGenerate tournament heatmap std")
    else:
        MPI.COMM_WORLD.Send( [total_rewards_normalized, MPI.DOUBLE], dest=0, tag=rank )
        logger.info(f"\nRank {rank} normalized total_reward sent")
    
    # calculate processing time
    end_time = MPI.Wtime()
    if rank == 0:
        logger.info(f"\nProcessing time: {end_time-start_time}")


def heatmap_plot(total_rewards_normalized, checkpoint, args, ranks=None, opt='default'):
    if opt == 'default':
        if ranks == None:
            raise Exception(f'No rank info for default heatmap plot!')
    # convert to dataframe for plotting
    heat_data_P1 = total_rewards_normalized[:,:,0]
    heat_data_P2 = total_rewards_normalized[:,:,1]
    P1_ticks = ["gen "+str(x) for x in checkpoint]
    P2_ticks = ["gen "+str(x) for x in checkpoint]
    df_P1 = pd.DataFrame(data=heat_data_P1, index=P1_ticks, columns=P2_ticks)
    df_P2 = pd.DataFrame(data=heat_data_P2, index=P1_ticks, columns=P2_ticks)

    # set title & labels
    P1_title = 'default_title_P1'
    P2_title = 'default_title_P2'
    xlabel = 'default_xlabel'
    ylabel = 'default_ylabel'
    P1_savename = 'default_savename_P1.png'
    P2_savename = 'default_savename_P2.png'
    if opt == 'default':
        P1_title = f"{args.env_name} row player average score with {args.games} gameplays"
        P2_title = f"{args.env_name} column player average score with {args.games} gameplays"
        xlabel = f"Checkpoints of seed {ranks[1]} for column player"
        ylabel = f"Checkpoints of seed {ranks[0]} for row player"
        P1_savename = f'./plot_tournament/{args.env_name}_{ranks[0]}vs{ranks[1]}_P1_{args.arange[0]}.{args.arange[1]}.{args.arange[2]}_g{args.games}.png'
        P2_savename = f'./plot_tournament/{args.env_name}_{ranks[0]}vs{ranks[1]}_P2_{args.arange[0]}.{args.arange[1]}.{args.arange[2]}_g{args.games}.png'
    elif opt == 'avg':
        P1_title = f"{args.env_name} row player average score with {args.games} gameplays across {args.population} seeds"
        P2_title = f"{args.env_name} column player average score with {args.games} gameplays across {args.population} seeds"
        xlabel = f"Checkpoints for column player"
        ylabel = f"Checkpoints for row player"
        P1_savename = f'./plot_tournament/{args.env_name}_avg_P1_{args.arange[0]}.{args.arange[1]}.{args.arange[2]}_g{args.games}.png'
        P2_savename = f'./plot_tournament/{args.env_name}_avg_P2_{args.arange[0]}.{args.arange[1]}.{args.arange[2]}_g{args.games}.png'
    elif opt == 'std':
        P1_title = f"{args.env_name} row player average score std with {args.games} gameplays across {args.population} seeds"
        P2_title = f"{args.env_name} column player average score std with {args.games} gameplays across {args.population} seeds"
        xlabel = f"Checkpoints for column player"
        ylabel = f"Checkpoints for row player"
        P1_savename = f'./plot_tournament/{args.env_name}_std_P1_{args.arange[0]}.{args.arange[1]}.{args.arange[2]}_g{args.games}.png'
        P2_savename = f'./plot_tournament/{args.env_name}_std_P2_{args.arange[0]}.{args.arange[1]}.{args.arange[2]}_g{args.games}.png'

    # generate heat plot
    sns.set(rc={'figure.figsize':(15,12)})
    ax = sns.heatmap(df_P1, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap=args.cmap)
    ax.set_title(P1_title.title(),fontsize=25)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    # ax.xaxis.tick_top()
    plt.xticks(rotation=45)
    fig = ax.get_figure()
    fig.savefig(P1_savename) 
    fig.clf()

    ax = sns.heatmap(df_P2, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap=args.cmap)
    ax.set_title(P2_title.title(),fontsize=25)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    # ax.xaxis.tick_top()
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