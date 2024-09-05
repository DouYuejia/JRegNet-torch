



import sys
import numpy as np

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_eager_execution()



from nets import *
from cfgs import *
from data import *
from clip_ops.clip_ops import *
from trainer import *

print(("Setting: %s"%(sys.argv[1])))
setting = sys.argv[1]

if setting == "aaa3":
    cfg = aaa3.cfg
    Net = unit_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer.Trainer

elif setting == "Integrated_A":
    cfg = Integrated_A.cfg
    Net = unit_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer.Trainer

else:
    print("None selected")
    sys.exit(0)
    
net = Net(cfg)
generator = Generator(cfg, 'test')
m = Trainer(cfg, "test", net, clip_op_lambda)
m.test(generator)
