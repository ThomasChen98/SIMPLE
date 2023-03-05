### Calculate influence level index for given environment
### Author: Yuxin Chen
### Date: Feb 24, 2023

### Sample usage
# sudo docker-compose exec app python3 influenceLevelIndex.py -e tictactoe -g 100 -a 1 60 3
# sudo docker-compose exec app python3 influenceLevelIndex.py -e tictactoe -g 100 -a 1 60 3 -l tictactoe_1_60_3_g100.npz

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
from scipy.special import rel_entr

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
    if args.load != None and not os.path.exists(os.path.join(config.RIDGELINEDIR, args.load)):
        raise Exception(f'{args.load} does not exist!')
    elif args.load != None:
        logger.info(f'\nLoading {args.load} data and plot ridgeline...')
        loaded = np.load(os.path.join(config.RIDGELINEDIR, args.load))
        reward_prob_dist = loaded['reward_prob_dist']
        kl_divergence = loaded['kl_divergence']
        influence_level_index = loaded['influence_level_index']
        checkpoint = loaded['checkpoint']
        total_rewards = loaded['total_rewards']
        ridgeline_plot(reward_prob_dist, kl_divergence, influence_level_index, checkpoint, total_rewards, args)
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
    total_rewards = np.zeros((policy_num, args.games, env.n_players))

    # probability distribution
    reward_prob_dist = np.zeros((policy_num, 3)) # P(R|theta_j) = [P(-1), P(0), P(1)] for ego player
    marginal_prob_dist = np.zeros(3) # P(R) = [P(-1), P(0), P(1)] for ego player

    # agents
    agents = []
    
    # play games
    logger.info(f'\nPlaying {args.games} games for each of {policy_num} policies...')
    for j in range(policy_num):
        # set up pairing agents
        agents.append(Agent('P1', models[-1])) # ego agent always uses the best policy
        agents.append(Agent('P2', models[j]))
        logger.debug(f'Agent j policy {j+1}: P1 = {model_list[-1]}: {agents[0]}, P2 = {model_list[j]}: {agents[1]}')

        for game in range(args.games):
            # reset env
            obs = env.reset()
            done = False
            logger.debug(f'Agent j policy {j+1} #{game+1} start')

            # shuffle player order
            players = agents[:]
            logger.debug(f'Agent j policy {j+1} #{game+1} P1 = {players[0]}, P2 = {players[1]}')
            if args.randomise_players:
                random.shuffle(players)

            # debug info
            for index, player in enumerate(players):
                logger.debug(f'Agent j policy {j+1} #{game+1}: Player {index+1} = {player.name}')

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
                    total_rewards[j][game][roster[player.name]] += r

                # pause after each turn to wait for user to continue
                if args.cont:
                    input('Press any key to continue')
                
                env.render()

                logger.debug(f"Agent j policy {j+1} #{game+1} total reward so far: {total_rewards[j][game]}")
            
            logger.debug(f"Agent j policy {j+1} #{game+1} finished total reward: {total_rewards[j][game]}")

            # calculate probability distribution
            if total_rewards[j][game][0] == -1:
                reward_prob_dist[j][0] += 1 # reward = -1
            elif total_rewards[j][game][0] == 0:
                reward_prob_dist[j][1] += 1 # reward = 0
            else:
                reward_prob_dist[j][2] += 1
            logger.debug(f"Agent j policy {j+1} #{game+1} reward prob dist: {reward_prob_dist[j]}")
        
        # convert reward_prob_dist to probability density
        reward_prob_dist[j] /= sum(reward_prob_dist[j])
        logger.info(f"Agent j policy {j+1} finished")
        logger.debug(f"Agent j policy {j+1} reward prob density: {reward_prob_dist[j]}")

        # calculate marginal prob dist
        marginal_prob_dist += reward_prob_dist[j]/policy_num # assume uniform policy sampling
        logger.debug(f"Agent j policy {j+1} marginal prob density so far: {marginal_prob_dist}")

        # reset agents
        agents = []

    logger.debug(f"Agent j policy {j+1} marginal prob density: {marginal_prob_dist}")

    env.close()

    # calculate influence level index
    kl_divergence = np.zeros(policy_num)
    influence_level_index = 0
    for j in range(policy_num): 
        kl_divergence[j] = sum(rel_entr(reward_prob_dist[j],marginal_prob_dist))
        logger.debug(f"KL divergence between P(R|theta{j+1}) and P(R): {kl_divergence[j]}")
        influence_level_index += kl_divergence[j]/policy_num # assume uniform policy sampling
    print(f'Influence Level Index of {args.env_name} approximated by given {policy_num} policies: {influence_level_index}')

    # save data
    save_name = f'./ridgeline/{args.env_name}_{args.arange[0]}_{args.arange[1]}_{args.arange[2]}_g{args.games}'
    np.savez_compressed(save_name, reward_prob_dist=reward_prob_dist, kl_divergence=kl_divergence,\
                        influence_level_index=influence_level_index, total_rewards=total_rewards,\
                        checkpoint=checkpoint)

    # plot
    ridgeline_plot(reward_prob_dist, kl_divergence, influence_level_index, checkpoint, total_rewards, args)


def ridgeline_plot(reward_prob_dist, kl_divergence, influence_level_index, checkpoint, total_rewards, args):
    # convert to dataframe for plotting
    model_name = ["model "+str(x) for x in checkpoint]
    policy_num = len(checkpoint)
    # density dataframe for kdeplot & histplot
    density_df = pd.DataFrame()
    for j in range(policy_num):
        temp = pd.DataFrame(total_rewards[j],columns=['P1 Reward','P2 Reward'])
        temp.insert(0, 'Model', model_name[j])
        temp.insert(3, 'KL Divergence', kl_divergence[j])
        density_df = density_df.append(temp, ignore_index=True)
    
    # plot
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(policy_num, rot=-.25, light=.7)
    g = sns.FacetGrid(density_df, row="Model", hue="Model", aspect=25, height=.4, palette=pal)

    # Draw the densities in a few steps
    # g.map(sns.kdeplot, "P1 Reward", bw_adjust=.3, clip_on=False, fill=True, alpha=1, linewidth=1)
    # g.map(sns.kdeplot, "P1 Reward", bw_adjust=.3, clip_on=False, color="w", lw=1.5)

    g.map(sns.histplot, "P1 Reward", bins=[-1.25,-0.75,-0.25,0.25,0.75,1.25], element="poly", fill=True, shrink=.8, alpha=1)
    g.map(sns.histplot, "P1 Reward", bins=[-1.25,-0.75,-0.25,0.25,0.75,1.25], element="poly", fill=False, color='w')

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(label, "P1 Reward")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.7)

    # set xticks
    plt.xticks([-1, 0, 1])
    plt.suptitle(f"{args.env_name} player 1 reward distribution with {args.games} gameplays\n influence_level_index={influence_level_index:.5f}".title())

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    g.savefig(f'./ridgeline/P1_{args.env_name}_{args.arange[0]}_{args.arange[1]}_{args.arange[2]}_g{args.games}.png')
    

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