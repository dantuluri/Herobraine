"""
config.py -- The main configuratio nstore for the behavioral cloning demo.
"""
GYM_RESOLUTION = (96*2, 128*2)
MALMO_IP = ('127.0.0.1', 10000)

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
    
SHARD_SIZE = 5000
RECORD_INTERVAL = 1.0/10.0
EPISODE_LENGTH = 1200
RANDOM_PERTURBATION_LEN = 10
PETURB_SPACE = {"turn -1", "turn 1", "pitch -1", "pitch 1"}

# HYPERPARAMETERS
SMALLEST_FEATURE_MAP = 9
FC_SIZES =[768, 128]
LEARNING_RATE = 1e-3
DROPOUT=0.5
BATCH_SIZE = 32
PIXEL_RESOLUTION = 255.0

