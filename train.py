import json
import logging
import os
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score

# Source Configurations
train_source_model = 'uniform'
infer_source_model = 'uniform'
max_k_train = False
x_lambda = 10
x_min_max = [1 / 32768, 1]

# Display Configurations
print_train_progress = True
print_stats = True
plot_the_summary = False

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def comp(Y, A):
    return ((Y == 0.0)@A == 0).astype(float)


def get_batch_data(mini_batch, k, n, source, A, **kwargs):
    # TODO: kwargs will contain above config params
    x = np.zeros((mini_batch, n))
    for i in range(n):
        sparsity = np.random.randint(0, k+1)
        positions = np.random.choice(np.arange(n), sparsity, replace=False)
        if source == 'poisson':
            x[i, positions] = np.random.poisson(x_lambda, sparsity)
        elif source == 'uniform':
            x[i, positions] = np.random.uniform(x_min_max[0], x_min_max[1], sparsity)
        else:
            raise Exception("Unsupported source")

    y = np.matmul(x, np.transpose(A))
    x_comp = comp(y, A)
    return x, y, x_comp


# def get_mini_batch(mini_batch, k, n, source, A):
#     if not max_k_train:
#         x_batch = get_batch_data(mini_batch, k, n, source, A)
#     else:
#         x_batch = get_batch_data(mini_batch // k, 1, n, source, A)
#         if k > 1:
#             for d in range(2, k + 1):
#                 x_batch = np.vstack((x_batch, get_batch_data(mini_batch // k, d, n, source, A)))
#
#     return x_batch


def initialize_nn(m, n, layers, lr):
    # Init graph
    tf.reset_default_graph()

    # Input and Output
    x = tf.placeholder(dtype=tf.float32, shape=[None, n], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, m], name='y')
    x_comp = tf.placeholder(dtype=tf.float32, shape=[None, n], name='x_comp')

    labels = tf.cast(tf.math.greater(x, 0), tf.float32)

    fc = tf.contrib.layers.fully_connected
    dropout = tf.contrib.layers.dropout
    y1 = tf.concat([y, x_comp], axis=1)
    for layer in layers:
        y1 = dropout(fc(y1, layer, activation_fn=tf.nn.relu), 1.0)

    x_est = dropout(fc(y1, n, activation_fn=tf.keras.activations.linear), 1.0)

    # Loss
    mse = tf.losses.mean_squared_error(labels, x_est)
    cross_entropy_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=x_est, pos_weight=4))
    add = tf.reduce_mean(tf.nn.relu(x - x_est))
    t = 0.0
    loss = cross_entropy_loss + t * add

    # Optimizer
    optimiser = tf.train.AdamOptimizer(
        # learning_rate=lr
    ).minimize(loss)

    nn_arch = {'x': x, 'y': y, 'optimiser': optimiser, 'loss': loss, 'mse': mse, 'add': add, 'x_est': x_est, 'x_comp': x_comp}

    return nn_arch


def train_nn(sess, nn_arch, max_epochs, A, k, n, log_idx, mini_batch):
    losses = []
    # writer = tf.summary.FileWriter('./graphs', sess.graph) # Writes graph to tensorboard

    for epoch in range(max_epochs):
        x_batch, y_batch, x_comp = get_batch_data(mini_batch, k, n, train_source_model, A)

        _, loss, mse, add = sess.run([nn_arch['optimiser'], nn_arch['loss'], nn_arch['mse'], nn_arch['add']],
                                     feed_dict={nn_arch['x']: x_batch,
                                                nn_arch['y']: y_batch,
                                                nn_arch['x_comp']: x_comp})
        if epoch % log_idx == 0:
            if print_train_progress:
                logging.info(f'epoch: {epoch},\tloss: {loss},\tmse: {mse},\tadd: {add}')
            losses.append([epoch, loss])

    train_results = {'losses': losses}
    return train_results


def infer_nn_stats(sess, nn_arch, d_max, n, A, test_batch):
    # Performance Stats (Inference)
    eps = 10 ** -5
    perf_dict = {}
    for d in range(1, d_max + 1):
        x_test, y_test, x_comp = get_batch_data(test_batch, d, n, infer_source_model, A)
        [x_estimate] = sess.run([nn_arch['x_est']],
                                feed_dict={nn_arch['x']: x_test,
                                           nn_arch['y']: y_test,
                                           nn_arch['x_comp']: x_comp})
        y_true = np.where(x_test > eps, 1, 0).ravel()
        y_pred = np.where(x_estimate > 0.5, 1, 0).ravel()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc = roc_auc_score(y_true, x_estimate.ravel())

        perf_dict[d] = [precision, recall, specificity, auc]

    return perf_dict


def infer_nn_estimate(sess, nn_arch, k, n, test_batch):
    # Estimation (Inference)
    x_test, y_test, x_comp = get_batch_data(test_batch, k, n, infer_source_model, A)
    [x_estimate] = sess.run([nn_arch['x_est']],
                            feed_dict={nn_arch['x']: x_test,
                                       nn_arch['y']: y_test,
                                       nn_arch['x_comp']: x_comp})
    # x_estimate_clustered = np.zeros(x_estimate.shape)
    # for i in range(test_batch):
    #     clusters, centroids = kmeans1d.cluster(x_estimate[i], 2)
    #     x_estimate_clustered[i] = x_estimate[i] * np.array(clusters)

    infer_results = {'x_test': x_test, 'x_estimate': x_estimate, 'x_estimate_clustered': x_estimate}
    return infer_results


def plot_summary(tr_results, inf_results, display_batch):
    # Analysis and Plots
    losses = np.array(tr_results['losses'])

    fig, ax = plt.subplots(3, 2)
    ax[0, 0].plot(losses[:, 0], losses[:, 1], label='Training Loss')
    ax[0, 0].set_xlabel('Epochs')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].legend()

    # ax0 = plt.subplot(3, 2, (3,5))
    # ax0.imshow(s_mat, cmap='Greys_r')
    # ax0.set_title("Learnt Sensing Matrix (A)")
    # ax0.set_xlabel("White = 1, Black = 0")

    ax1 = plt.subplot(3, 2, 2)
    ax1.imshow(inf_results['x_test'][list(range(display_batch)), :], cmap='Greys_r')
    ax1.set_title("True x samples")
    ax1.set_xlabel("White = Non-zero, Black = Zero")

    ax2 = plt.subplot(3, 2, 4)
    ax2.imshow(inf_results['x_estimate'][list(range(display_batch)), :], cmap='Greys_r')
    ax2.set_title("Estimated x samples")
    ax2.set_xlabel("White = High, Black = Low")

    ax3 = plt.subplot(3, 2, 6)
    ax3.imshow(inf_results['x_estimate_clustered'][list(range(display_batch)), :], cmap='Greys_r')
    ax3.set_title("Estimated x samples after clustering")
    ax3.set_xlabel("White = High, Black = Low")

    plt.tight_layout()
    plt.show()


def save_learnt_sensing_mat(sess, nn_arch, beta_inf):
    s_matrix = np.transpose(sess.run(nn_arch['A'], feed_dict={nn_arch['beta_factor']: [[beta_inf]]}))
    min_col = min(sum(s_matrix, 0))
    min_row = min(sum(s_matrix, 1))
    print('Verify if min counts are > 0: min_col_count: ', min_col, '\tmin_row_count: ', min_row)

    if min_col > 0 and min_row > 0:
        np.savetxt('learnt_matrix.txt', s_matrix, fmt='%i', delimiter=' ')
        return s_matrix
    else:
        raise Exception("Learnt matrix has either all-zero row/ col."
                        " If it is all-zero col. increase mini-batch size! All-zero row needs more deeper inspection")


def print_sensing_matrix(s_mat):
    np.set_printoptions(linewidth=np.inf, precision=4, suppress=True)
    print(s_mat)


def jsr_pipeline(max_epochs, tr_batch, test_batch, m, n, k, A, log_idx, d_max, layers, disp_batch, lr):
    nn_arch = initialize_nn(m, n, layers, lr)
    num_cores = 1
    num_CPU = 1
    num_GPU = 0
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores,
                            allow_soft_placement=True,
                            device_count={'CPU': num_CPU,
                                          'GPU': num_GPU}
                            )
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        train_perf = train_nn(sess, nn_arch, max_epochs, A, k, n, log_idx, tr_batch)
        infer_stats = infer_nn_stats(sess, nn_arch, d_max, n, A, test_batch)

        if plot_the_summary:
            estimates = infer_nn_estimate(sess, nn_arch, k, n, disp_batch)
            plot_summary(train_perf, estimates, disp_batch)

    if print_stats:
        key_list = list(infer_stats.keys())
        key_list.sort()
        logging.info("Sparsity\tPrecision\tRecall (Sensitivity)\tSpecificity\tAUC")
        for key in key_list:
            logging.info('\t%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f' % (key, *[round(val, 4) for val in infer_stats[key]]))


if __name__ == "__main__":
    params = {}
    ap = ArgumentParser()
    # ap.add_argument("--data_path", type=str, default="./data/squad/squad.pkl")
    ap.add_argument("--logfile", type=str, default="./logs/tapestry.log")
    ap.add_argument("--epochs", type=int, default=50000)
    ap.add_argument("--sparsity", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    av = ap.parse_args()

    params.update(vars(av))
    num_items = 105
    num_tests = 45
    sparsity = av.sparsity
    # For N-layered decoder network, we will have len(decoder_hidden_layers) = N-1
    decoder_hidden_layers = [105, 105]
    learn_rate = 0.001
    mini_batch_size = 1024
    max_num_epochs = av.epochs
    log_index = 1000
    test_batch_size = 4096

    display_batch_size = 5
    sigma = 0.1
    d_max_stats = 20
    A = np.loadtxt("./optimized_M_45_285_kirkman.txt", dtype='i', delimiter=' ')
    A = A[:, :105]
    seed = av.seed

    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger = logging.getLogger('logger')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s', filename=av.logfile,
                        filemode='a')
    logging.info(f'training code for the params:\n{json.dumps(params, indent=4)}')
    jsr_pipeline(max_num_epochs, mini_batch_size, test_batch_size,
                 num_tests, num_items, sparsity, A, log_index,
                 d_max_stats, decoder_hidden_layers,
                 display_batch_size, learn_rate)
