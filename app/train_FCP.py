# sudo docker-compose exec app python3 train_FCP.py -r -e tictactoe -tt 2e7 -t 0.5 -p 3 -tn 5
# sudo docker-compose exec app mpirun -np 5 python3 train_FCP.py -r -e tictactoe -tt 2e7 -t 0.5

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import argparse
import time
import multiprocessing as mp
from mpi4py import MPI

from stable_baselines.ppo1 import PPO1

from stable_baselines.common import set_global_seeds
from stable_baselines import logger

from utils.callbacks import FictitiousCoPlayCallback
from utils.files import reset_logs, reset_models_PP
from utils.register import get_environment
from utils.fictitiouscoplay import fictitiouscoplay_wrapper

import config

def main(threadID, args):
  rank = MPI.COMM_WORLD.Get_rank()
  start_time = MPI.Wtime()

  # setup logs and models
  model_dir = os.path.join(config.MODELDIR, args.env_name, f'thread_{threadID}')
  temp_model_dir = os.path.join(config.TMPMODELDIR, f'thread_{threadID}')

  if rank == 0:
    try:
      os.makedirs(model_dir)
      os.makedirs(temp_model_dir)
    except:
      pass
    reset_logs(threadID)
    time.sleep(5)
    if args.reset:
      reset_models_PP(model_dir, args.env_name, threadID)
    logger.configure(os.path.join(config.LOGDIR, f'thread_{threadID}'))
  else:
    logger.configure(format_strs=[])

  if args.debug:
    logger.set_level(config.DEBUG)
  else:
    time.sleep(5)
    logger.set_level(config.INFO)

  workerseed = args.seed + 10 * threadID + 10000 * MPI.COMM_WORLD.Get_rank()
  set_global_seeds(workerseed)
  logger.info(f'\n+++Thread {threadID}+++ Workerseed: {workerseed} Population size: {args.population_size}')

  logger.info(f'\n+++Thread {threadID}+++ Setting up the FCP training environment opponents...')
  base_env = get_environment(args.env_name)
  env = fictitiouscoplay_wrapper(base_env)(threadID = threadID, population=args.population_size, opponent_type = args.opponent_type, verbose = args.verbose)
  env.seed(workerseed)

  params = {'gamma':args.gamma,
    'timesteps_per_actorbatch':args.timesteps_per_actorbatch,
    'clip_param':args.clip_param,
    'entcoeff':args.entcoeff,
    'optim_epochs':args.optim_epochs,
    'optim_stepsize':args.optim_stepsize,
    'optim_batchsize':args.optim_batchsize,
    'lam':args.lam,
    'adam_epsilon':args.adam_epsilon,
    'schedule':'linear',
    'verbose':1,
    'tensorboard_log':os.path.join(config.LOGDIR, f'thread_{threadID}')
  }

  time.sleep(5) # allow time for the base model to be saved out when the environment is created

  if args.reset or not os.path.exists(os.path.join(model_dir, f'best_model_{threadID}.zip')):
    logger.info(f'\n+++Thread {threadID}+++ Loading the base PPO agent to train...')
    threadID_str = str(threadID).zfill(2)
    model = PPO1.load(os.path.join(model_dir, f'_base_{threadID_str}.zip'), env, **params)
  else:
    logger.info(f'\n+++Thread {threadID}+++ Loading the best_model.zip PPO agent to continue training...')
    model = PPO1.load(os.path.join(model_dir, f'best_model_{threadID}.zip'), env, **params)

  #Callbacks
  logger.info(f'\n+++Thread {threadID}+++ Setting up the FCP evaluation environment opponents...')
  callback_args = {
    'eval_env': fictitiouscoplay_wrapper(base_env)(threadID=threadID, population=args.population_size, opponent_type = args.opponent_type, verbose = args.verbose),
    'best_model_save_path' : temp_model_dir,
    'log_path' : os.path.join(config.LOGDIR, f'thread_{threadID}'),
    'eval_freq' : args.eval_freq,
    'n_eval_episodes' : args.n_eval_episodes,
    'deterministic' : False,
    'render' : True,
    'verbose' : 0
  }
      
  # Evaluate the agent against previous versions
  eval_callback = FictitiousCoPlayCallback(threadID, args.opponent_type, args.threshold, args.env_name, **callback_args)

  logger.info(f'\n+++Thread {threadID}+++ Setup complete - commencing learning...\n')

  model.learn(total_timesteps=int(args.total_timesteps), callback=[eval_callback], reset_num_timesteps = False, tb_log_name="tb")

  # calculate processing time
  end_time = MPI.Wtime()
  logger.info(f"\n+++Thread {threadID}+++ Processing time: {end_time-start_time}")

  env.close()
  del env


if __name__ =="__main__":
    # Setup argparse to show defaults on help
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)


    parser.add_argument("--reset", "-r", action = 'store_true', default = False
                , help="Start retraining the model from scratch")
    parser.add_argument("--opponent_type", "-o", type = str, default = 'best'
                , help="best / mostly_best / random / base / rules - the type of opponent to train against")
    parser.add_argument("--debug", "-d", action = 'store_true', default = False
                , help="Debug logging")
    parser.add_argument("--verbose", "-v", action = 'store_true', default = False
                , help="Show observation in debug output")
    parser.add_argument("--rules", "-ru", action = 'store_true', default = False
                , help="Evaluate on a ruled-based agent")
    parser.add_argument("--best", "-b", action = 'store_true', default = False
                , help="Uses best moves when evaluating agent against rules-based agent")
    parser.add_argument("--env_name", "-e", type = str, default = 'tictactoe'
                , help="Which gym environment to train in: tictactoe, connect4, sushigo, butterfly, geschenkt, frouge")
    parser.add_argument("--seed", "-s",  type = int, default = 17
                , help="Random seed")

    parser.add_argument("--eval_freq", "-ef",  type = int, default = 10240
                , help="How many timesteps should each actor contribute before the agent is evaluated?")
    parser.add_argument("--n_eval_episodes", "-ne",  type = int, default = 100
                , help="How many episodes should each actor contirbute to the evaluation of the agent")
    parser.add_argument("--threshold", "-t",  type = float, default = 0.2
                , help="What score must the agent achieve during evaluation to 'beat' the previous version?")
    parser.add_argument("--gamma", "-g",  type = float, default = 0.99
                , help="The value of gamma in PPO")
    parser.add_argument("--timesteps_per_actorbatch", "-tpa",  type = int, default = 1024
                , help="How many timesteps should each actor contribute to the batch?")
    parser.add_argument("--clip_param", "-c",  type = float, default = 0.2
                , help="The clip paramater in PPO")
    parser.add_argument("--entcoeff", "-ent",  type = float, default = 0.1
                , help="The entropy coefficient in PPO")

    parser.add_argument("--optim_epochs", "-oe",  type = int, default = 4
                , help="The number of epoch to train the PPO agent per batch")
    parser.add_argument("--optim_stepsize", "-os",  type = float, default = 0.0003
                , help="The step size for the PPO optimiser")
    parser.add_argument("--optim_batchsize", "-ob",  type = int, default = 1024
                , help="The minibatch size in the PPO optimiser")
    parser.add_argument("--total_timesteps", "-tt",  type = float, default = 2e7
                , help="Total env steps for training")
                
    parser.add_argument("--lam", "-l",  type = float, default = 0.95
                , help="The value of lambda in PPO")
    parser.add_argument("--adam_epsilon", "-a",  type = float, default = 1e-05
                , help="The value of epsilon in the Adam optimiser")

    parser.add_argument("--thread_num", "-tn",  type = int, default = 1
                , help="Number of threads run in parallel")
    parser.add_argument("--population_size", "-p",  type = int, default = 1
                , help="Population size for FCP")

    # Extract args
    args = parser.parse_args()

    # creating thread
    process_list =  [mp.Process(target=main, args=(i, args)) for i in range(args.thread_num)]

    for process in process_list:
        process.start()
        
    for process in process_list:
        process.join()
    
    # main(0,args)