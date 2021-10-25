
from __future__ import print_function

import os
import random
import sys
import time
from optparse import OptionParser

import numpy as np
import tensorflow as tf
from scipy import spatial

from agent import DeepAgent
from deep_sea_treasure import DeepSeaTreasure
from utils import *


parser = OptionParser()
parser.add_option(
    "-l",
    "--algorithm",
    dest="alg",
    choices=["scal", "mo", "mn", "cond", "uvfa", "random", "naive"],
    default="cond",
    help="Architecture type, one of 'scal','mo','meta','cond'")
parser.add_option(
    "-d",
    "--dupe",
    dest="dupe",
    default="CN",
    choices=["none", "CN", "CN-UVFA", "CN-ACTIVE"],
    help="Extra training")
parser.add_option(
    "--end_e",
    dest="end_e",
    default="0.01",
    help="Final epsilon value",
    type=float)
parser.add_option(
    "--start_e",
    dest="start_e",
    default="0.1",
    help="start epsilon value",
    type=float)
parser.add_option(
    "-r", "--lr", dest="lr", default="0.02", help="learning rate", type=float)
parser.add_option(
    "--clipnorm", dest="clipnorm", default="1", help="clipnorm", type=float)
parser.add_option(
    "--mem-a", dest="mem_a", default="2.", help="memory error exponent", type=float)
parser.add_option(
    "--mem-e", dest="mem_e", default="0.01", help="error offset", type=float)
parser.add_option(
    "--clipvalue",
    dest="clipvalue",
    default="0.5",
    help="clipvalue",
    type=float)
parser.add_option(
    "--momentum", dest="momentum", default="0.9", help="momentum", type=float)
parser.add_option(
    "-u",
    "--update_period",
    dest="updates",
    default="4",
    help="Update interval",
    type=int)
parser.add_option(
    "--target-update",
    dest="target_update_interval",
    default="150",
    help="Target update interval",
    type=int)
parser.add_option(
    "-f",
    "--frame-skip",
    dest="frame_skip",
    default="1",
    help="Frame skip",
    type=int)
parser.add_option(
    "--sample-size",
    dest="sample_size",
    default="64",
    help="Sample batch size",
    type=int)
parser.add_option(
    "-g",
    "--discount",
    dest="discount",
    default="0.95",
    help="Discount factor",
    type=float)
parser.add_option("--scale", dest="scale", default=1,
                help="Scaling", type=float)
parser.add_option("--anneal-steps",
                dest="steps", default=10000, help="steps",  type=int)
parser.add_option("-x", "--extra", dest="extra", default="")
parser.add_option(
    "-c", "--mode", dest="mode", choices=["regular", "sparse"], default="regular")
parser.add_option(
    "-s", "--seed", dest="seed", default=None, help="Random Seed", type=int)
parser.add_option("-n", "--ner", dest='ner', default=True)


(options, args) = parser.parse_args()


# dst = gym.make('BountifulSeaTreasure-v1')
dst = DeepSeaTreasure(view=(5,5), full=True, scale=1)
obj_cnt = 2

all_weights = list(np.loadtxt("CN/regular_weights_dst"))
# all_weights = [np.array([0.41111111, 1-0.41111111])]
# all_weights = all_weights + [np.array([0.09832561, 1-0.09832561]) for i in range(steps)]
# print(all_weights[0])
# extra = "Attentive_1-lstm-{} clipN-{} clipV-{} attention-{} a-{} m-{} s-{}  e-{} d-{} x-{} {} p-{} fs-{} d-{} up-{} lr-{} e-{} p-{} m-{}-{}".format(
# options.lstm, options.clipnorm, options.clipvalue, options.attention,
# options.alg, options.mem, options.seed,
# options.end_e, options.dupe, options.extra, options.mode, options.reuse,
# options.frame_skip,
# np.round(options.discount, 4), options.updates,
# np.round(options.lr, 4),
# np.round(options.scale, 2), np.round(options.steps, 2), np.round(options.mem_a, 2), np.round(options.mem_e, 2))
extra = "AP_6-regular"

agent = DeepAgent(
    range(4), #range(ACTION_COUNT). e.g. range(6)
    obj_cnt,        
    options.steps,
    sample_size=options.sample_size,
    weights=None,
    discount=options.discount,
    learning_rate=options.lr,
    target_update_interval=options.target_update_interval,
    alg=options.alg,
    frame_skip=options.frame_skip,
    start_e=options.start_e,
    end_e=options.end_e,
    update_interval=options.updates,
    mem_a=options.mem_a,
    mem_e=options.mem_e,
    extra=extra,
    clipnorm=options.clipnorm,
    clipvalue=options.clipvalue,
    momentum=options.momentum,
    scale=options.scale,
    dupe=None if options.dupe == "none" else options.dupe,
    ner=options.ner)

steps_per_weight = 5000 if options.mode == "sparse" else 1
log_file = open('CN/output/logs/rewards_{}'.format(extra), 'w', 1)
agent.train(dst, log_file, options.steps, all_weights, steps_per_weight, options.steps*10)