import sys, os
import numpy as np
# import tensorflow as tf
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '../rts/game_MC'))
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import minirts
import tqdm

from rts import Loader
from rlpytorch import ArgsProvider
from datetime import datetime
from elf import GCWrapper


cnt_predict = 0
cnt_forward = 0
cnt_project = 0


def actor(batch):
    '''
    import pdb
    pdb.set_trace()
    pickle.dump(utils_elf.to_numpy(sel), open("tmp%d.bin" % k, "wb"), protocol=2)
    '''
    return dict(a=[0]*batch["s"].size(1))


def reduced_predict(batch):
    global cnt_predict
    cnt_predict += 1
    # print("in reduced_predict, cnt_predict = %d" % cnt_predict)


def reduced_forward(batch):
    global cnt_forward
    cnt_forward += 1
    # print("in reduced_forward, cnt_forward = %d" % cnt_forward)


def reduced_project(batch):
    global cnt_project
    cnt_project += 1
    # print("in reduced_project, cnt_project = %d" % cnt_project)


# ======================= load environment =======================
co = minirts.ContextOptions()  # need to check the candidate keys
co.num_games = 20
co.T = 1
co.wait_per_group = 1
co.verbose_comm = 1

opt = minirts.PythonOptions()

GC = minirts.GameContext(co, opt)

batch_descriptions = {
    "actor": dict(
        batchsize=128,
        input=dict(T=1, keys=set(["s", "a", "last_r", "terminal"])),
        reply=dict(T=1, keys=set(["rv", "pi", "V", "a"]))
    ),
    "reduced_predict": dict(
        batchsize=128,
        input=dict(T=1, keys=set(["s", "a", "last_r", "terminal"])),
        reply=dict(T=1, keys=set(["rv", "pi", "V", "a"]))
    ),
    "reduced_forward": dict(
        batchsize=128,
        input=dict(T=1, keys=set(["s", "a", "last_r", "terminal"])),
        reply=dict(T=1, keys=set(["rv", "pi", "V", "a"]))
    ),
    "reduced_project": dict(
        batchsize=128,
        input=dict(T=1, keys=set(["s", "a", "last_r", "terminal"])),
        reply=dict(T=1, keys=set(["rv", "pi", "V", "a"]))
    )
}

GC = GCWrapper(GC, co, batch_descriptions, use_numpy=True, params=GC.GetParams())

GC.reg_callback("actor", actor)
GC.reg_callback("reduced_predict", reduced_predict)
GC.reg_callback("reduced_forward", reduced_forward)
GC.reg_callback("reduced_project", reduced_project)

GC.Start()

while True:
    GC.Run()

GC.Stop()

