import os
import numpy as np
import random

from utils.files import load_model_with_id, load_all_models, get_best_model_name, get_model_length
from utils.agents import Agent

from mpi4py import MPI

from stable_baselines import logger

def selfplay_wrapper(env):
    class SelfPlayEnv(env):
        # wrapper over the normal single player env, but loads the best self play model
        def __init__(self, threadID, opponent_type, verbose):
            super(SelfPlayEnv, self).__init__(verbose)
            self.threadID = threadID
            self.rank = MPI.COMM_WORLD.Get_rank()
            self.opponent_type = opponent_type
            self.opponent_model = load_all_models(self, self.threadID)[-1]
            self.best_model_name = get_best_model_name(self.name, self.threadID)
            self.model_length = get_model_length(self.name, self.threadID) # store the number of saved best models

        def setup_opponents(self):
            # incremental load of new model
            if get_model_length(self.name, self.threadID) > self.model_length:
                self.model_length = get_model_length(self.name, self.threadID)
                best_model_name = get_best_model_name(self.name, self.threadID)
                logger.info(f'+++Thread {self.threadID}+++ New opponent model: {best_model_name}, previous opponent model = {self.best_model_name}')
                self.opponent_model = load_model_with_id(self, best_model_name, self.threadID)
                self.best_model_name = best_model_name

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
            super(SelfPlayEnv, self).reset()
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
                observation, reward, done, _ = super(SelfPlayEnv, self).step(action)
                logger.debug(f'+++Thread {self.threadID}+++ Rewards: {reward}')
                logger.debug(f'+++Thread {self.threadID}+++ Done: {done}')
                if done:
                    break

            return observation, reward, done, None


        def step(self, action):
            self.render()
            observation, reward, done, _ = super(SelfPlayEnv, self).step(action)
            logger.debug(f'+++Thread {self.threadID}+++ Action played by agent: {action}')
            logger.debug(f'+++Thread {self.threadID}+++ Rewards: {reward}')
            logger.debug(f'+++Thread {self.threadID}+++ Done: {done}')

            if not done:
                package = self.continue_game()
                if package[0] is not None:
                    observation, reward, done, _ = package


            agent_reward = reward[self.agent_player_num]
            logger.debug(f'\n+++Thread {self.threadID}+++ Reward To Agent: {agent_reward}')

            if done:
                self.render()

            return observation, agent_reward, done, {}

    return SelfPlayEnv