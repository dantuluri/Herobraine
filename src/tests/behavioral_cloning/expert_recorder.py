"""
The expert recorder.
"""
import argparse
import keyboard

import gym
import gym_minecraft

from IPython import embed

BINDINGS = [
    ({'w': 'move 1',
    "s": 'move -1'}, "move 0"),
    ({"a": 'strafe -1',
    "d": 'strafe 1'}, "strafe 0"),
    ({ "k": 'pitch 1',
    "i": 'pitch -1'}, "pitch 0"),
    ({"l": 'turn 1',
    "j": 'turn -1'}, "turn 0"),
    ({'space': 'jump 1'}, "jump 0"),
    ({"n": 'attack 1'}, "attack 0"),
    ({"m": 'use 1'}, "use 0")]

def get_options():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data_directory', type=str,
        help="The main datastore for this particular expert.")

    args = parser.parse_args()

    return args


def run_recorder(opts):
    """
    Runs the main recorder by binding certain discrete actions to keys.
    """
    ddir = opts.data_directory

    record_history = [] # The state action history buffer.

    env = gym.make('MinecraftDefaultWorld1-v0')
    env.init(
        start_minecraft=None,
        client_pool=[('127.0.0.1', 10000)],
        continuous_discrete = True,
        videoResolution=(128*2,96*2),
        add_noop_command=True)

    ##############
    # BIND KEYS  #1
    ##############

    keyboard.unhook_all()
    keys_pressed = {}
    action = ""

    def keyboard_hook(event):
        """
        The key manager for interaction with minecraft.
        Allow sfor simultaneous execution of movement 
        """
        nonlocal action, keys_pressed
        if event.event_type is keyboard.KEY_DOWN:
            keys_pressed[event.name] = True
        else:
            del keys_pressed[event.name]

        actions_to_process = []
        for kmap, default in BINDINGS:
            pressed = [x for x in kmap if x in keys_pressed]
            if len(pressed) > 1 or len(pressed) == 0:
                actions_to_process.append(default)
            else:
                actions_to_process.append(kmap[pressed[0]])

        action = "\n".join(actions_to_process)
        print(action)



    keyboard.hook(keyboard_hook)

    env.reset()

    done = False
    while not done:
        env.render()
        # action = env.action_space.sample()
        if keys_pressed and action:
            obs, reward, done, info = env.step(action)
        else:
            env.step(action)


    keyboard.unhook(keyboard_hook)

if __name__ == "__main__":
    opts = get_options()
    run_recorder(opts)