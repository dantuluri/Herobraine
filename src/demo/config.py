"""
config.py -- The main configuration store for the behavioral cloning demo.
"""
GYM_RESOLUTION = (96*2, 128*2)
MALMO_IP = ('127.0.0.1', 10001)

# DONOT CHANGE THE ORDER HERE.
# IT WILL INVALIDATE OLD DATA!
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

BINDINGS_NEW = [{ 'w':'move 1', 'a':'strafe -1', 's':'move -1', 'd':'strafe 1', ' ':'jump 1',
     "k": 'pitch 1', 'i': 'pitch -1', "l": 'turn 1', "j": 'turn -1', "n": 'attack 1',
     "m": 'use 1'},
     { 'w':'move 0', 'a':'strafe 0', 's':'move 0', 'd':'strafe 0', ' ':'jump 0',
     "k": 'pitch 0', 'i': 'pitch 0', "l": 'turn 0', "j": 'turn 0', "n": 'attack 0',
     "m": 'use 0'}]

    
SHARD_SIZE = 5000
RECORD_INTERVAL = 1.0/10.0
EPISODE_LENGTH = 120000
DATA_DIR = "./"

# HYPERPARAMETERS
SMALLEST_FEATURE_MAP = 9
FC_SIZES =[768, 400]
LEARNING_RATE = 1e-4
DROPOUT=0
BATCH_SIZE = 32
PIXEL_RESOLUTION = 255.0
LSTM_HIDDEN = 128
MAX_SEQ_LEN = 20
