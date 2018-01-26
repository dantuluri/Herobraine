"""
Cloner.py -- The main program for cloning expert data from minecraft.
    1. Loads data from some bc_data store
    2. Starts a training loop and trains the model
    3. Starts a demonstration loop and runs a snapshot of  the model
       for a specified number of frames.
"""
import tensorflow as tf
import numpy as np
import os
import argparse
import threading
from bidict import bidict

import time
import gym
import minecraft_py
import gym_minecraft
import keyboard

from config import(
    GYM_RESOLUTION,
    MALMO_IP,
    BINDINGS, 
    BATCH_SIZE,
    EPISODE_LENGTH,
    RECORD_INTERVAL)

# Import hyperparameters.
from agent import Agent

def get_options():
    """
    Parse the command line arguments for the cloner.
    """
    parser = argparse.ArgumentParser(description='Clone some expert data..')
    parser.add_argument('bc_data', type=str,
        help="The main datastore for this particular expert.")

    return  parser.parse_args()


def run_training(coord, agent, bc_data_dir, action_map):
    """
    Runs training for the agent.
    """
    # Load the file store. 
    # In the future (TODO) move this to a seperate thread.
    states, actions = [], []
    shards = [x for x in os.listdir(bc_data_dir) if x.endswith('.npy')]
    print("Processing shards: {}".format(shards))
    for shard in shards:
        shard_path = os.path.join(bc_data_dir, shard)
        with open(shard_path, 'rb') as f:
            data = np.load(f)
            shard_states, unprocessed_actions = zip(*data)
            print(np.asarray(shard_states).shape)
            
            # Process actions
            split_actions = [a.split("\n") for a in unprocessed_actions]
            shard_actions = [
                [actionbd.inv[v] for actionbd, v in zip(action_map, saction)] for saction in split_actions
            ]
            # Add the shard to the dataset
            states.extend(shard_states)
            actions.extend(shard_actions)

    states = np.asarray(states, dtype=np.uint8)
    actions = np.asarray(actions, dtype=np.float32)
    print("Processed with {} pairs".format(len(states)))

    # Train the agent from the file store
    for tick in range(100000):
        # Get a random batch using an increasing batch size schedule
        batch_index = np.random.choice(len(states), BATCH_SIZE*2**(min(tick//1000, 2)))
        state_batch, action_batch = states[batch_index], actions[batch_index]

        # Run the training procedure
        loss = agent.train(state_batch, action_batch)
        if tick % 10 == 0:
            print("Loss @ {}: {}".format(tick, loss))

        # Persist the agent occasionally in the same directory.
        # Stop in case there is 
        if coord.should_stop():
            # Save potentially.
            break





def run_demonstrations(coord, agent, action_map):
    """
    Runs the main demonstration for the agent.
    """


    # keyboard.unhook_all()
    # keys_pressed = {}
    # restart = False

    # def keyboard_hook(event):
    #     """
    #     The key manager for interaction with minecraft.
    #     Allow sfor simultaneous execution of movement 
    #     """
    #     nonlocal keys_pressed, restart
    #     if event.event_type is keyboard.KEY_DOWN:
    #         keys_pressed[event.name] = True
    #     else:   
    #         if 'r' in keys_pressed: restart = True
    #         if event.name in keys_pressed:
    #             del keys_pressed[event.name]

    # keyboard.hook(keyboard_hook)

    while not coord.should_stop():
        failed = True
        while failed:
            try:
                env = gym.make('MinecraftDefaultWorld1-v0')
                env.init(
                    start_minecraft=None,
                    client_pool=[MALMO_IP],
                    continuous_discrete = True,
                    videoResolution=(GYM_RESOLUTION[1], GYM_RESOLUTION[0]),
                    add_noop_command=True)
                failed = False
            except Exception as e:
                pass
        # Create the environment with specified arguments
        last_obs = env.reset()
        last_action = ''
        last_action_time = time.time()


        for tick in range(EPISODE_LENGTH):
            env.render()
            cur_time = time.time()
            
            # # Restart
            # if restart:
            #     restart = False
            #     break

            # If the record interval has passed
            if cur_time - last_action_time > RECORD_INTERVAL:
                # Get the agents action
                action_index = agent.act(np.asarray(last_obs))
                action = "\n".join([samap[i] for samap, i in zip(action_map, action_index)])

                # Step the environment.
                obs, reward, done, info = env.step(action)
                print(action.split("\n"))

                # Potnetially back up obs sobs pair.
                # If not done.

                last_action_time = cur_time
                last_action = action
                last_obs = obs
            else:
                last_obs,  reward, done, info  = env.step(last_action)




def run_main(opts):
    # Define the action space. (length of key bindings plus null action)
    action_space = [len(d) + 1 for d,_ in BINDINGS]
    action_map = []
    for act_dict, no_act in BINDINGS:
        action_map.append(
            bidict(dict(
                list(
                    enumerate(sorted(act_dict.values())))
                 + [(len(act_dict), no_act )])))

    # Define the state space.
    state_space = list(GYM_RESOLUTION) + [3] # 3 color channels

    # Create a model
    sess = tf.InteractiveSession()
    agent = Agent(state_space, action_space, sess)
    sess.run(tf.global_variables_initializer())

    # Start the training thread with a coordinator.
    coord = tf.train.Coordinator()
    training_thread = threading.Thread(
        target=run_training, args=(coord, agent, opts.bc_data, action_map))
    training_thread.start()

    # Begin performing demonstrations.
    time.sleep(20)
    run_demonstrations(coord, agent, action_map)
    coord.join([training_thread])

if __name__ == "__main__":
    # Parse arguments
    opts = get_options()
    # Start the main thread.
    run_main(opts)