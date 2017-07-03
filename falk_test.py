import tensorflow.python.platform

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 9})
import os
import re
import sys
import pickle
import time
from tqdm import tqdm

# Probably number of input parameters (for x1,y1,...,xn,yn => FEATURE_LEN = n)
FEATURE_LEN = 30
# Don't worry about this. Increase for less performance and more accuracy
NUM_LSTM_LAYERS = 300
# Model diverging? Set learning rate lower and NUM_LSTM_LAYERS higher
LEARNING_RATE = 1e-4
# L2 tries to make the weights equal. Just don't have this too high (can be zero if you're just fooling around)
L2_REG_PARAM = 1e-6
# Leave this either at RMSPropOptimizer or AdamOptimizer. They are different optimization algorithms
OPTIMIZER = tf.train.RMSPropOptimizer
# Number of future frames to take into account
VALUES_TO_PREDICT = 5

# No tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.app.flags.DEFINE_string('train', None,
		'File containing the training data.')
tf.app.flags.DEFINE_integer('num_epochs', 10,
		'Number of passes over the training data.')
tf.app.flags.DEFINE_integer('num_frames', 5,
                'Number of previous frames to take into account')
FLAGS = tf.app.flags.FLAGS

def extract_data(filename):
    _fname = re.search('\/.*', filename).group(0)[1:]

    # if os.path.isfile("data_np/%s_fvecs.mp"%_fname):
        # sys.stdout.write("Loading %s... " %filename)
        # sys.stdout.flush()

        # shapeinfo = pickle.load(open("data_np/%s.shapeinfo"%_fname, "r"))
        # fvecs = np.memmap("data_np/%s_fvecs.mp"%_fname, dtype="float32", mode="r", shape=shapeinfo)

        # print("Done")
        # return fvecs

    sys.stdout.write("Exporting %s... " %filename)
    sys.stdout.flush()
    # Arrays to hold the labels and feature vectors.
    fvecs = []

    def pad_line(vec):
        while len(vec) < FEATURE_LEN:
            # insert 0s at the beginning
            vec.insert(0, 0)

        while len(vec) > FEATURE_LEN:
            vec = vec[:-1]

        return vec

    for i,line in enumerate(file(filename)):
        line_data = line.split(" ")

        points = []
        for i in range(len(line_data)):
            if i == 0:
                continue

            i_ = i - 1
            if i_ % 3 == 2:
                x = float(line_data[i-2])
                y = float(line_data[i-1])

                points.append((x,y))

        fvecs.append(np.transpose(pad_line(points)))

    # Convert the array of float arrays into a numpy int matrix.
    fvecs_np = np.array(fvecs).astype(np.float32)

    print("Done")

    pickle.dump( fvecs_np.shape, open("data_np/%s.shapeinfo"%_fname, "wb"))

    fvecs = np.memmap("data_np/%s_fvecs.mp"%_fname, dtype="float32", mode="w+", shape=fvecs_np.shape)

    fvecs[:] = fvecs_np[:]
    del fvecs_np

    return fvecs

def lstm_model(lstm_input, output_shape):
    NUM_FRAMES = FLAGS.num_frames


    weights = tf.get_variable('lstm-weights', shape=[NUM_LSTM_LAYERS, 5], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

    biases = tf.get_variable('lstm-biases', shape=[5], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

    def normalize(__x):
        _max = tf.reshape(tf.reduce_max(__x, axis=1), [-1,1])
        _min = tf.reshape(tf.reduce_min(__x, axis=1), [-1,1])

        return (__x - _min) / (_max - _min)

    # get a list of all frames
    x = tf.split(lstm_input, NUM_FRAMES, axis=0)
    for i in range(len(x)):
        x[i] = tf.reshape(x[i], [1, 2 * FEATURE_LEN])

    tf.identity(x, name="x")

    lstm_cell = tf.contrib.rnn.LSTMCell(NUM_LSTM_LAYERS)

    # get lstm output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    tf.identity(outputs[-1], name="outputs")

    lstm_out = tf.matmul(outputs[-1], weights) + biases
    lstm_out = tf.reshape(lstm_out, output_shape, name="lstm_out")

    return lstm_out, weights, biases

def mdn_model(mdn_input, output_seq, weights, biases):
    mux, varx, muy, vary, rho = tf.split(mdn_input, 5, axis=0)

    # apply exp function to varx and vary, since we don't want them to be zero
    varx = tf.exp(varx)
    vary = tf.exp(vary)

    # apply the tanh function to rho (ref: graves paper)
    rho = tf.nn.tanh(rho)

    # get the names of all variables, so we can use them for evaluation
    tf.identity(mux, name="mux")
    tf.identity(varx, name="varx")
    tf.identity(muy, name="muy")
    tf.identity(vary, name="vary")
    tf.identity(rho, name="rho")

    # get the future frame into a shape that is useful to us
    output_x, output_y = tf.split(output_seq, 2, axis=0)
    output_x = tf.transpose(output_x)
    output_y = tf.transpose(output_y)

    # this is gonna be your custom normal distribution
    # for now, it is a simple 2d normal dist
    def tf_normal(out_x, out_y, mx, my, sx, sy, rho):
        z1 = tf.square(out_x - mx) / tf.square(sx)
        z2 = tf.square(out_y - my) / tf.square(sy)
        z3 = (2*rho * (out_x - mx)*(out_y - my)) / (sx * sy)

        Z = z1 + z2 - z3
        tf.identity(Z, name="Z")

        n1 = tf.exp(-Z / (2*(1-tf.square(rho))))
        n2 = 1/(2*np.pi*sx*sy*tf.sqrt(1-tf.square(rho)))

        N = n1 * n2

        return N

    # the loss function could be anything at this point
    def get_loss():
        res = tf_normal(output_x, output_y, mux, muy, varx, vary, rho)
        res = tf.reduce_sum(res, axis=1, keep_dims=True)
        res = -tf.log(res)
        return tf.reduce_sum(res, name="loss")

    loss = get_loss()

    # this is the l2 regularization, which tries to have equal weights
    l2_loss = tf.nn.l2_loss(weights) * L2_REG_PARAM
    l2_loss = tf.identity(l2_loss, name="l2_loss")
    loss = tf.add(loss, l2_loss, name="total_loss")

    # the function we will call over and over again to train the model
    trainer = OPTIMIZER(LEARNING_RATE).minimize(loss, var_list=[weights, biases])

    return trainer

def model(input_seq, output_seq):
    NUM_FRAMES = FLAGS.num_frames

    # output 4 parameters: mux, varx, muy, vary
    output_shape = [5]
    lstm_out, weights, biases = lstm_model(input_seq, output_shape)

    trainer = mdn_model(lstm_out, output_seq, weights, biases)

    return trainer

def run_test(sess, data):
    num_frames = FLAGS.num_frames

    x = tf.get_default_graph().get_tensor_by_name("ph_x:0")

    # the variables that we need to plot the trained normal distribution
    mux = tf.get_default_graph().get_tensor_by_name("mux:0")
    muy = tf.get_default_graph().get_tensor_by_name("muy:0")
    varx = tf.get_default_graph().get_tensor_by_name("varx:0")
    vary = tf.get_default_graph().get_tensor_by_name("vary:0")
    rho = tf.get_default_graph().get_tensor_by_name("rho:0")

    def eval_thing(acc, d, loop_iter):
        # get the values for our distribution from our model (given the input frames)
        _mx, _my, _vx, _vy, _rho = sess.run([mux, muy, varx, vary, rho], {x: acc})

        # Just the plotting range
        x_lim = (0, 10)
        y_lim = (-5,5)

        # The resolution of the resulting image
        resolution = 1000

        # ranges for the heatmap
        x_range = np.linspace(x_lim[0], x_lim[1], resolution)
        y_range = np.linspace(y_lim[0], y_lim[1], resolution)

        def np_normal(mx, my, sx, sy, rho):
            z1 = np.square(mx/sx)
            z2 = np.square(my/sy)
            z3 = 2*rho*mx*my/(sx*sy)

            Z = z1+z2-z3

            n1 = np.exp(-Z / (2*(1-np.square(rho))))
            n2 = 1/(2*np.pi*sx*sy*np.sqrt(1-np.square(rho)))

            return n1*n2

        # create the heatmap
        [X,Y] = np.meshgrid(x_range, -y_range)
        Z = np_normal(_mx - X, _my - Y, _vx, _vy, _rho)

        plt.figure(figsize=(15,10))
        plt.scatter(acc[-1][0], acc[-1][1], c="w", alpha = 0.5)
        plt.imshow(Z, interpolation="Nearest", extent = [x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
        plt.colorbar()
        plt.show()

        # save the plot into your folder
        # plt.savefig("figures/eval/%s_%i.png"%(time.strftime("%m%d-%M%H%S"), loop_iter))

    # We create an accumulator to get hold of a number of frames we need
    accumulator = []
    for j, d in enumerate(data):
        accumulator.append(d)
        accumulator = accumulator[-num_frames:]

        if len(accumulator) >= num_frames:
            eval_thing(accumulator, d, j)

def main(argv=None):
    train_data_filename = FLAGS.train
    num_frames = FLAGS.num_frames
    num_epochs = FLAGS.num_epochs

    train_data = extract_data(train_data_filename)

    # we have x as the input data and y as the output data
    x = tf.placeholder(tf.float32, shape=[num_frames, 2, FEATURE_LEN], name="ph_x")
    y = tf.placeholder(tf.float32, shape=[2, VALUES_TO_PREDICT], name="ph_y")

    # the trainer function that we will call in our training loop
    trainer = model(x,y)

    # get losses so we can plot them during training
    loss = tf.get_default_graph().get_tensor_by_name("loss:0")
    l2_loss = tf.get_default_graph().get_tensor_by_name("l2_loss:0")
    total_loss = tf.get_default_graph().get_tensor_by_name("total_loss:0")

    # get and initialize the tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # from all the data in the ram, take a slice to train the model on
    def get_frames(start):
        start = start % (len(train_data) - num_frames - VALUES_TO_PREDICT+1)

        # len of returned list is num_frames + 1 since the last element is the result
        f_x = train_data[start:start+num_frames]
        f_y = train_data[start+num_frames+VALUES_TO_PREDICT-1]
        f_y = f_y[:,-VALUES_TO_PREDICT:]

        return f_x, f_y

    # This is what we will be iterating over
    maxrange = len(train_data) * num_epochs
    bar = tqdm(range(maxrange))

    # Using this to plot some things
    losses = []
    l2_losses = []
    total_losses = []
    mean_loss_acc = []
    logged_vars = {}

    graph = tf.get_default_graph()
    plt.ion()
    for i in bar:
        try:
            # get the frames for input and for output
            frames_x, frames_y = get_frames(i)

            # Train the model
            feeder = {x: frames_x, y: frames_y}
            _, _l, _l2_l, _tl = sess.run([ trainer, loss, l2_loss, total_loss ], feeder)

            # Plot losses, do some logging...
            losses.append(_l)
            l2_losses.append(_l2_l)
            total_losses.append(_tl)

            t = len(train_data) - num_frames - VALUES_TO_PREDICT+1

            # plot the loss every x frames
            if i%500 == 0:
                plt.clf()
                plt.plot(losses[:], label="loss")
                plt.plot(total_losses[:], label="total_loss")
                plt.plot(l2_losses[:], label="l2_loss")

                for j in range(t):
                    plt.plot([k*t+j for k,_ in enumerate(total_losses[j::t])], total_losses[j::t], alpha=.2)

                plt.xlim(max(0, len(losses)-1000), len(losses))
                plt.legend()
                plt.tight_layout()
                plt.pause(0.001)

        except KeyboardInterrupt:
            break

    plt.ioff()
    plt.close()

    run_test(sess, train_data)

if __name__ == '__main__':
    tf.app.run()
