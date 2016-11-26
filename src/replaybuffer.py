import lmdb
import numpy as np

lmdb_file = "/home/william/fucks"
lmdb_env = lmdb.open(lmdb_file)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()