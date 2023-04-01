import os
import numpy as np
import random

from utils.files import load_SP_model_with_id, load_all_models, get_best_model_name, get_random_SP_model_name, get_model_length, get_opponent_length, get_current_opponent_name_id
from utils.agents import Agent

from mpi4py import MPI

import config

from stable_baselines import logger

def fictitiouscoplay_wrapper(env):
    class FictitiousCoPlayEnv(env):
        # wrapper over the normal single player env, but loads the best self play model
        def __init__(self, threadID, population, opponent_type, verbose, update = True):
            super(FictitiousCoPlayEnv, self).__init__(verbose)
            self.threadID = threadID
            self.rank = MPI.COMM_WORLD.Get_rank()
            self.population = population
            self.opponent_model = load_all_models(self, self.threadID)[-1]
            self.opponent_length = get_opponent_length(self.name, self.threadID)
            self.opponent_id = -1
            self.best_model_name = get_best_model_name(self.name, self.threadID)
            self.model_length = get_model_length(self.name, self.threadID) # store the number of saved best models

        def setup_opponents(self):
            # incremental load of new model
            if get_model_length(self.name, self.threadID) > self.model_length:
                self.model_length = get_model_length(self.name, self.threadID)
                if get_opponent_length(self.name, self.threadID) == self.opponent_length: # get new opponent id
                    self.opponent_id = random.randint(0, self.population-1)
                    logger.info(f'\n+++Thread {self.threadID}+++ New opponent id: {self.opponent_id}')

                    best_model_name = get_random_SP_model_name(self.name, self.opponent_id)

                    logger.info(f'+++Thread {self.threadID}+++ New opponent model: {best_model_name}, previous opponent model = {self.best_model_name}')
                    self.opponent_model = load_SP_model_with_id(self, best_model_name, self.opponent_id)
                    self.best_model_name = best_model_name

                    # saving new opponent model
                    opponent_num = get_opponent_length(self.name, self.threadID)
                    opponent_str = str(opponent_num).zfill(5)+best_model_name
                    logger.info(f'+++Thread {self.threadID}+++ Saving new opponent model: {opponent_str}')
                    with open(os.path.join(config.MODELDIR, self.name, f'thread_{self.threadID}','opponents.txt'), "a") as f:
                        f.write('\n')
                        f.write(opponent_str)
                    self.opponent_length = get_opponent_length(self.name, self.threadID)
                else: # evaluation, keep opponent id
                    best_model_name, opponent_id = get_current_opponent_name_id(self.name, self.threadID)
                    self.opponent_id = opponent_id
                    logger.info(f'\n+++Thread {self.threadID}+++ Evaluate on current opponent id: {self.opponent_id}')

                    logger.info(f'+++Thread {self.threadID}+++ Evaluate New opponent model: {best_model_name}, previous opponent model = {self.best_model_name}')
                    self.opponent_model = load_SP_model_with_id(self, best_model_name, self.opponent_id)
                    self.best_model_name = best_model_name

                    self.opponent_length = get_opponent_length(self.name, self.threadID)
            
            self.opponent_agent = Agent('ppo_opponent', self.opponent_model)  

            self.agent_player_num = np.random.choice(self.n_players)
            self.agents = [self.opponent_agent] * self.n_players
            self.agents[self.agent_player_num] = None
            try:
                #if self.players is defined on the base environment
                logger.debug(f'+++Thread {self.threadID}+++ Agent plays as Player {self.players[self.agent_player_num].id}')
            except:
                pass


        def reset(self):
            super(FictitiousCoPlayEnv, self).reset()
            self.setup_opponents()

            if self.current_player_num != self.agent_player_num:   
                self.continue_game()

            return self.observation

        @property
        def current_agent(self):
            return self.agents[self.current_player_num]

        def continue_game(self):
            observation = None
            reward = None
            done = None

            while self.current_player_num != self.agent_player_num:
                self.render()
                action = self.current_agent.choose_action(self, choose_best_action = False, mask_invalid_actions = False)
                observation, reward, done, _ = super(FictitiousCoPlayEnv, self).step(action)
                logger.debug(f'+++Thread {self.threadID} Rank {self.rank}+++ Rewards: {reward}')
                logger.debug(f'+++Thread {self.threadID} Rank {self.rank}+++ Done: {done}')
                if done:
                    break

            return observation, reward, done, None


        def step(self, action):
            self.render()
            observation, reward, done, _ = super(FictitiousCoPlayEnv, self).step(action)
            logger.debug(f'+++Thread {self.threadID} Rank {self.rank}+++ Action played by agent: {action}')
            logger.debug(f'+++Thread {self.threadID} Rank {self.rank}+++ Rewards: {reward}')
            logger.debug(f'+++Thread {self.threadID} Rank {self.rank}+++ Done: {done}')

            if not done:
                package = self.continue_game()
                if package[0] is not None:
                    observation, reward, done, _ = package


            agent_reward = reward[self.agent_player_num]
            logger.debug(f'\n+++Thread {self.threadID} Rank {self.rank}+++ Reward To Agent: {agent_reward}')

            if done:
                self.render()

            return observation, agent_reward, done, {}


    return FictitiousCoPlayEnv