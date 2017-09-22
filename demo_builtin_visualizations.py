from __future__ import print_function
import numpy as np
import datetime
from tensorboardX import SummaryWriter



################################################################################
### HELPFUNS AND ALIASES
################################################################################

def make_timestamp(prefix="", suffix=""):
    return prefix+"{:%d_%b_%Y_%Hh%Mm%Ss}".format(datetime.datetime.now())+suffix
rand = np.random.uniform
randint = np.random.randint
gaussian = np.random.normal

################################################################################
### ADDING ELEMENTS TO TENSORBOARD: SCALARS, TEXT, IMAGES...
################################################################################

def demo_scalar(logger, numsteps=1000):
    r = rand(2, 20)*3
    for i in range(0, numsteps):
        logger.add_scalar("xsinx", i*np.sin(i/r), i)
        logger.add_scalar("xcosx", i*np.cos(i/r), i)

def demo_text(logger, numsteps=1000):
    for i in range(0, numsteps, numsteps/10):
        a = "a"*randint(1, 10)+"ardvark"
        p = "pika"*randint(1,3)+"ch"+"u"*randint(1,10)
        logger.add_text("digger animal", a, i)
        logger.add_text("famous PoKeMon", p, i)

def demo_histogram(logger, numsteps=1000):
    rvar = rand(1, 10)
    for i in range(0, numsteps):
        dist_a = gaussian(-50*numsteps+100*i, i*rvar, 100)
        dist_b = gaussian(50*numsteps-100*i, numsteps*rvar, 100)
        logger.add_histogram("distribution A", dist_a, i)
        logger.add_histogram("distribution B", np.append(dist_a, dist_b), i)

# writer.add_image('Image', x, n_iter) 
# writer.add_audio('myAudio', x, n_iter)
# TODO: demo fns for the remaining logger methods
# writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))
################################################################################
### MAIN: INSTANTIATE LOGGER AND RUN THE DEMOS
################################################################################

TOTAL_RUNS = 2
NUMSTEPS = 100
for i in range(TOTAL_RUNS): # simulate different runs
    print("rendering run no.", i+1, "of", TOTAL_RUNS, "(may take a while...)")
    LOGDIR = make_timestamp("logs/", "run_"+str(i+1))
    LOGGER = SummaryWriter(LOGDIR)
    # add the data
    demo_scalar(LOGGER, NUMSTEPS)
    demo_text(LOGGER, NUMSTEPS)
    demo_histogram(LOGGER, NUMSTEPS)
    # finally flush and close the file stream. Apart from this explicit flush,
    # flushing is automatically performed asynch (at least every 120 secs, see
    # docstring for the FileWriter class), so information could get lost if
    # this isn't done before the logger is destroyed at process end.
    LOGGER.close()
