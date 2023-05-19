import numpy as np
import tensorflow as tf

def calculate_sigma_x(O_mat,samples,log_fxn):
    numsamples, N = np.shape(samples.numpy())
    samples_log_probs_fxn = log_fxn(samples)
    samples = tf.cast(samples,dtype=tf.float32)
    samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], N, axis=2)
    samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped + tf.transpose(O_mat)[tf.newaxis, :, :], 2) # numsamples, n_spins, n_spins
    samples_tiled_flipped = tf.transpose(samples_tiled_flipped,perm=[0,2,1]) 
    samples_tiled_flipped = tf.cast(tf.reshape(samples_tiled_flipped,(numsamples*N,N)),dtype=tf.int64) # every one sample becomes N samples each with 1 spin flip
    log_probs_flipped = log_fxn(samples_tiled_flipped)
    log_probs_flipped = tf.reshape(log_probs_flipped,(numsamples,N))
    log_prob_ratio = tf.math.exp(log_probs_flipped - samples_log_probs_fxn[:, tf.newaxis])
    sigma_xs_all_sites = log_prob_ratio
    sigma_xs = tf.reduce_mean(sigma_xs_all_sites,axis=1) # should now have one value for each of the samples
    return sigma_xs, sigma_xs_all_sites

def calculate_sigxi_sigxj(O_mat,samples,log_fxn):
    numsamples, N = np.shape(samples.numpy())
    samples_log_psi_fxn = log_fxn(samples)
    samples = tf.cast(samples,dtype=tf.float32)
    #flip 1
    samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], N, axis=2)
    samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped + tf.transpose(O_mat)[tf.newaxis, :, :], 2) # numsamples, n_spins, n_spins
    samples_tiled_flipped = tf.transpose(samples_tiled_flipped,perm=[0,2,1]) 
    samples_tiled_flipped = tf.reshape(samples_tiled_flipped,(numsamples*N,N)) # every one sample becomes N samples each with 1 spin flip
    #flip 2
    samples_tiled_flipped2 = tf.repeat(samples_tiled_flipped[:, :, tf.newaxis], N, axis=2)
    samples_tiled_flipped2 = tf.math.mod(samples_tiled_flipped2 + tf.transpose(O_mat)[tf.newaxis, :, :], 2) # numsamples, n_spins, n_spins
    samples_tiled_flipped2 = tf.transpose(samples_tiled_flipped2,perm=[0,2,1]) 
    samples_tiled_flipped2 = tf.reshape(samples_tiled_flipped2,(numsamples*N*N,N)) # every one sample becomes N samples each with 1 more spin flip
    samples_tiled_flipped2 = tf.cast(samples_tiled_flipped2,dtype=tf.int64)
    log_psi_flipped = log_fxn(samples_tiled_flipped2)
    log_psi_flipped = tf.reshape(log_psi_flipped,(numsamples,N*N))
    log_psi_ratios = tf.math.exp(log_psi_flipped - samples_log_psi_fxn[:, tf.newaxis])
    return log_psi_ratios