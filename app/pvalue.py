### Generate p-value heatmap between agents trained by different methods
### Author: Yuxin Chen
### Date: Apr 20, 2023

### Sample usage
# sudo docker-compose exec app mpirun -np 25 python3 pvalue.py -e tictactoe -r -g 100 -ld data
# sudo docker-compose exec app python3 pvalue.py -e tictactoe -g 100 -l TTT/tictactoe_std_g100.npz

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
import scipy

from stable_baselines import logger
from stable_baselines.common import set_global_seeds

from mpi4py import MPI

from utils.files import load_all_best_models
from utils.register import get_environment
from utils.agents import Agent

import config


def main(args):
    # if load previous data, directly plot the heatmap
    if args.load != None and not os.path.exists(os.path.join(config.PVALUEDIR, args.load)):
        raise Exception(f'{args.load} does not exist!')
    elif args.load != None:
        logger.info(f'\nLoading {args.load} data and plot heatmap...')
        loaded = np.load(os.path.join(config.PVALUEDIR, args.load))
        heatmap(loaded['pvalue'], loaded['total_count'],\
                 ['SP_TTT_20M_s8', 'PP_TTT_20M_s3', 'PP_TTT_20M_s5', 'FCP_TTT_20M_p3', 'FCP_TTT_20M_p5'], args)
        return
    
    start_time = MPI.Wtime()

    rank = MPI.COMM_WORLD.Get_rank()

    # setup logger
    logger.configure(config.PVALUELOGDIR)

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
    # policy_dir = ['SP_TTT_20M_s8', 'PP_TTT_20M_s3']
    policy_dir = ['SP_TTT_20M_s8', 'PP_TTT_20M_s3', 'PP_TTT_20M_s5',\
                  'FCP_TTT_20M_p3', 'FCP_TTT_20M_p5']
    
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
    winning_count = np.zeros((ego_policy_num, opp_policy_num, env.n_players)).astype(int)

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
                        if r == 1:
                            winning_count[i][j][roster[player.name]] += 1

                    # pause after each turn to wait for user to continue
                    if args.cont:
                        input('Press any key to continue')
                    
                    env.render()
                    
                    logger.debug(f"Gameplay {i}-{j} #{game} step: {winning_count[i][j]}")
            
                logger.debug(f"Gameplay {i}-{j} #{game} finished: {winning_count[i][j]}")
            
            logger.info(f"Gameplay {i}-{j} finished")

            # reset agents
            agents = []

    env.close()

    # stats for both players
    total_game = ego_policy_num * opp_policy_num * args.games
    wins_count = np.apply_over_axes(np.sum, winning_count, [0,1]).squeeze()
    loses_count = wins_count[::-1]
    draws_count = total_game - wins_count - loses_count
    total_count = np.vstack([wins_count,draws_count,loses_count,[total_game]*2])
    pvalue = np.zeros(2) # Null Hypothesis: P1 does not loses to P2 by 50% chance
    for i in range(2):
        pvalue[i] = scipy.stats.binom_test(wins_count[i]+draws_count[i], total_game, p=0.5, alternative='greater')

    logger.info(f"Total games {total_game}")
    logger.info(f"wins_count {wins_count[0]} -- {wins_count[1]}")
    logger.info(f"loses_count {loses_count[0]} -- {loses_count[1]}")
    logger.info(f"draws_count {draws_count[0]} -- {draws_count[1]}")
    logger.info(f"P-value {pvalue[0]} -- {pvalue[1]}")

    # plot total map
    if rank == 0:
        world_pvalue = np.zeros((len(policy_dir), len(policy_dir), env.n_players))
        world_total_count = np.zeros((len(policy_dir), len(policy_dir), 4, env.n_players))
        world_pvalue[0,0,:] = pvalue
        world_total_count[0,0,:,:] = total_count
        for i in range( 1, MPI.COMM_WORLD.Get_size() ):
            cur_pvalue = np.zeros(env.n_players)
            cur_total_count = np.zeros([4, env.n_players])
            MPI.COMM_WORLD.Recv( [cur_pvalue, MPI.DOUBLE], source=i, tag=i )
            logger.info(f"{i}th pvalue received")
            MPI.COMM_WORLD.Recv( [cur_total_count, MPI.DOUBLE], source=i, tag=i+100 )
            logger.info(f"{i}th total_count received")
            col = i//len(policy_dir)
            row = i%len(policy_dir)
            world_pvalue[col,row,:] = cur_pvalue
            world_total_count[col,row,:, :] = cur_total_count

        # save data
        save_name = f'./{config.PVALUEDIR}/{args.env_name}_g{args.games}'
        np.savez_compressed(save_name, pvalue=world_pvalue, total_count=world_total_count, policy_dir=policy_dir)

        # plot
        heatmap(world_pvalue, world_total_count, policy_dir, args)
        logger.info(f"\nGenerate heatmap")
    else:
        MPI.COMM_WORLD.Send( [pvalue, MPI.DOUBLE], dest=0, tag=rank )
        logger.info(f"\nRank {rank} p-pvalue sent")
        MPI.COMM_WORLD.Send( [total_count, MPI.DOUBLE], dest=0, tag=rank+100 )
        logger.info(f"\nRank {rank} total_count sent")
    
    # calculate processing time
    end_time = MPI.Wtime()
    logger.info(f"\nProcessing time: {end_time-start_time}")

def heatmap(pvalue, total_count, policy_dir, args):
    # convert to short name
    policy_name = []
    for f in policy_dir:
        name_vec = f.split('_')
        policy_name.append(f'{name_vec[0]}_{name_vec[-1]}')
    # convert to dataframe for plotting
    wins_perc = total_count[:,:,0,0]/total_count[:,:,-1,0]
    draws_perc = total_count[:,:,1,0]/total_count[:,:,-1,0]
    loses_perc = total_count[:,:,2,0]/total_count[:,:,-1,0]
    P1_ticks = [str(x) for x in policy_name]
    P2_ticks = [str(x) for x in policy_name]
    df_wins = pd.DataFrame(data=wins_perc, index=P1_ticks, columns=P2_ticks)
    df_draws = pd.DataFrame(data=draws_perc, index=P1_ticks, columns=P2_ticks)
    df_loses = pd.DataFrame(data=loses_perc, index=P1_ticks, columns=P2_ticks)

    # set title & labels
    title = [f"{args.env_name} row player wins percentage with {args.games} gameplays",\
             f"{args.env_name} row player draws percentage with {args.games} gameplays",\
             f"{args.env_name} row player loses percentage with {args.games} gameplays"]
    xlabel = f"Training methods for column player"
    ylabel = f"Training methods for row player"
    savename = [f'./{config.PVALUEDIR}/{args.env_name}_wins_g{args.games}.pdf',\
                f'./{config.PVALUEDIR}/{args.env_name}_draws_g{args.games}.pdf',\
                f'./{config.PVALUEDIR}/{args.env_name}_loses_g{args.games}.pdf']

    # generate heat plot
    df_list = [df_wins, df_draws, df_loses]
    for i in range(len(df_list)):
        sns.set(rc={'figure.figsize':(15, 13)})
        ax = sns.heatmap(df_list[i], annot=True, fmt=".2f", vmin=0, vmax=1, cmap=args.cmap, square = True, cbar_kws={"shrink": 1})
        ax.set_title(title[i].title(), fontsize=25, y=1.05)
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        # ax.xaxis.tick_top()
        plt.subplots_adjust(right=1)
        plt.xticks(rotation=45)
        fig = ax.get_figure()
        fig.savefig(savename[i]) 
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
  parser.add_argument("--cmap", "-cm", type = str, default = "flare"
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