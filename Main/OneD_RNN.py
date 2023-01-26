import numpy as np 
import tensorflow as tf 
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
import random

class OneD_RNN_wavefxn(tf.keras.Model):
    
    # Constructor
    def __init__(self, Lx, Ly, 
                 num_hidden, learning_rate,
                 seed=1234):
        
        super(OneD_RNN_wavefxn, self).__init__()
        print("OneD RNN used in previous Rydberg project.")

        """ PARAMETERS """
        self.Lx       = Lx              # Size along x
        self.Ly       = Ly              # Size along y
        self.N        = self.Lx * self.Ly
        self.nh       = num_hidden      # Number of hidden units in the RNN
        self.seed     = seed            # Seed of random number generator 
        self.K        = 2               # Dimension of the local Hilbert space
        self.lr       = learning_rate
        
        # Set the seed of the rng
        tf.random.set_seed(self.seed)

        # Optimizer
        self.optimizer = tf.optimizers.Adam(self.lr, epsilon=1e-8)

        # Build the model RNN
        # RNN layer: N -> nh
        self.rnn = tf.keras.layers.GRU(self.nh, kernel_initializer='glorot_uniform',
                                       kernel_regularizer = tf.keras.regularizers.l2(0.001),
                                       return_sequences = True,
                                       return_state = True,
                                       stateful = False)

        # Dense layer: nh - > K
        self.dense = tf.keras.layers.Dense(self.K, activation = tf.nn.softmax,
                                           kernel_regularizer = tf.keras.regularizers.l2(0.001))

        sample,_ = self.sample(1)
        self.logpsi(sample)
        self.trainable_variables_ = []
        self.trainable_variables_.extend(self.rnn.trainable_variables)
        self.trainable_variables_.extend(self.dense.trainable_variables)
        # Check that there are the correct amount of trainable variables
        self.variables_names = [v.name for v in self.trainable_variables_]
        sum_ = 0
        for k, v in zip(self.variables_names, self.trainable_variables_):
            v1 = tf.reshape(v, [-1])
            print(k, v1.shape)
            sum_ += v1.shape[0]
        print(f'The sum of params is {sum_}')
        
    @tf.function
    def sample(self,nsamples):
        inputs = 0.0*tf.one_hot(tf.zeros(shape=[nsamples,1],dtype=tf.int32),depth=self.K)
        hidden_state = tf.zeros(shape=[nsamples,self.nh])
        logP = tf.zeros(shape=[nsamples,],dtype=tf.float32)
        for j in range(self.N):
            rnn_output,hidden_state = self.rnn(inputs,initial_state=hidden_state)
            probs = self.dense(rnn_output)
            log_probs = tf.reshape(tf.math.log(1e-10+probs),[nsamples,self.K])
            sample = tf.random.categorical(log_probs,num_samples=1)
            if (j == 0):
                samples = tf.identity(sample)
            else:
                samples = tf.concat([samples,sample],axis=1)
            inputs = tf.one_hot(sample,depth=self.K)
            logP = logP+tf.reduce_sum(log_probs*tf.reshape(inputs,(nsamples,self.K)),axis=1)
        samples = tf.cast(samples,dtype=tf.int64)
        return samples,logP

    @tf.function
    def logpsi(self,samples):
        num_samples = samples.shape[0]
        data = tf.one_hot(samples[:,0:self.N-1],depth=self.K)
        # x0 = 0.0*tf.one_hot(tf.zeros(shape=[num_samples,1],dtype=tf.int32),depth=self.K) #initialization
        x0 = tf.zeros(shape=[num_samples,1,self.K],dtype=tf.float32)
        inputs = tf.concat([x0,data],axis=1)
        hidden_state = tf.zeros(shape=[num_samples,self.nh])
        rnn_output,_ = self.rnn(inputs,initial_state = hidden_state)
        probs        = self.dense(rnn_output)
        one_hot_samples = tf.one_hot(samples,depth=self.K,axis=2)
        log_probs   = tf.reduce_sum(tf.multiply(tf.math.log(1e-10+probs),one_hot_samples),axis=2)
        log_probs_return = 0.5 * tf.reduce_sum(log_probs, axis=1)
        return log_probs_return


class RNNWavefunction1D(tf.keras.Model):
    def __init__(self, Lx, Ly, 
                 num_hidden, learning_rate,
                 seed=1234):
        
        super(RNNWavefunction1D, self).__init__()
        print("OneD RNN adapted from Heisenberg-Kagome project.")

        """
            systemsize:             int
                                    number of sites
            num_units:              list of int
                                    number of num_units per RNN layer
            local_hilbert_space:    int
                                    size of the local hilbet space
                                    (related to the number of spins per site, which 
                                    is related to the lattice of interest)
            cell:                   a tensorflow RNN cell
            activation:  activation of the RNN cell
            seed:        pseudo-random number generator
            h_symmetries: bool
                         determines whether HAMILTONIAN symmetries are
                         enforced during the sampling step
            l_symmetries: bool
                         determines whether LATTICE symmetries are 
                         enforced during the sampling step
        """

        self.Lx       = Lx              # Size along x
        self.Ly       = Ly              # Size along y
        self.N        = self.Lx * self.Ly
        self.nh       = num_hidden      # Number of hidden units in the RNN
        self.local_hilbert_space = 2               # Dimension of the local Hilbert space
        self.seed     = seed
        self.lr        =learning_rate

        # Set the seed of the rng
        tf.random.set_seed(self.seed)

        # Optimizer
        self.optimizer = tf.optimizers.Adam(self.lr, epsilon=1e-8) 
        print(f"The input learning rate is {learning_rate}")
        print(f"The optimizer learning rate is {self.optimizer.lr.numpy()}")

        # Defining RNN cells with site-dependent parameters
        self.rnn = tf.keras.layers.GRU(self.nh, kernel_initializer='glorot_uniform',
                                       kernel_regularizer = tf.keras.regularizers.l2(0.001),
                                       return_sequences = True,
                                       return_state = True,
                                       stateful = False)
        self.dense = tf.keras.layers.Dense(2, activation=tf.nn.softmax, name='RNNWF_dense_0', dtype=tf.float32)

        sample,_ = self.sample(10)
        self.logpsi(sample)
        self.trainable_variables_ = []
        self.trainable_variables_.extend(self.rnn.trainable_variables)
        self.trainable_variables_.extend(self.dense.trainable_variables)
        # Check that there are the correct amount of trainable variables
        self.variables_names = [v.name for v in self.trainable_variables_]
        sum_ = 0
        for k, v in zip(self.variables_names, self.trainable_variables_):
            v1 = tf.reshape(v, [-1])
            print(k, v1.shape)
            sum_ += v1.shape[0]
        print(f'The sum of params is {sum_}')

    @tf.function
    def sample(self, numsamples):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:

            num_samples:      int
                             number of samples to be produced

            ------------------------------------------------------------------------
            Returns:         a tuple (samples,log-probs)

            samples:         tf.Tensor of shape (num_samples,systemsize)
                             the samples in integer encoding
            log-probs        tf.Tensor of shape (num_samples,)
                             the log-probability of each sample
        """
        samples = []
        probs = []

        inputs = tf.zeros((numsamples, 1, self.local_hilbert_space), dtype=tf.float32)  # Feed the table b in tf.
        rnn_state = self.rnn.get_initial_state(inputs)

        for n in range(self.N):
            rnn_output, rnn_state = self.rnn(inputs, rnn_state)
            output_prob = tf.squeeze(self.dense(rnn_output))
            sample_temp = tf.random.categorical(tf.math.log(output_prob), num_samples=1)
            sample_temp = tf.reshape(sample_temp,[-1,])
            probs.append(output_prob)
            samples.append(sample_temp)
            inputs = tf.one_hot(sample_temp, depth=self.local_hilbert_space, dtype=tf.float32)
            inputs = tf.reshape(inputs,(numsamples, 1, self.local_hilbert_space))

        samples = tf.stack(values=samples, axis=1)
        probs = tf.transpose(tf.stack(values=probs, axis=2), perm=[0, 2, 1])
        one_hot_samples = tf.one_hot(samples, depth=self.local_hilbert_space, dtype=tf.float32)
        log_probs = 0.5*tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs, one_hot_samples), axis=2)), axis=1)
        samples = tf.cast(samples, dtype=tf.float32)
        return samples, log_probs

    @tf.function
    def logpsi(self, samples):
        """
            calculate the log-probabilities of ```samples``
            ------------------------------------------------------------------------
            Parameters:

            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,system-size)
                             containing the input samples in integer encoding

            ------------------------------------------------------------------------
            Returns:
            log-probs        tf.Tensor of shape (number of samples,)
                             the log-probability of each sample
            """
        numsamples = samples.shape[0]
        probs = []
        samples = tf.cast(samples, dtype=tf.int32)
        inputs = tf.zeros((numsamples,1, self.local_hilbert_space), dtype=tf.float32)  # Feed the table b in tf.
        rnn_state = self.rnn.get_initial_state(inputs)

        # Logic in H-K version
        for n in range(self.N):
            rnn_output, rnn_state = self.rnn(inputs, rnn_state)
            output_prob = tf.squeeze(self.dense(rnn_output))
            probs.append(output_prob)
            inputs = tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples, begin=[0, n], size=[-1, 1]),
                                                      shape=[numsamples]), depth=self.local_hilbert_space,
                                           dtype=tf.float32), shape=[numsamples, self.local_hilbert_space])
            inputs = tf.reshape(inputs,(numsamples, 1, self.local_hilbert_space))

        probs = tf.transpose(tf.stack(values=probs, axis=2), perm=[0, 2, 1])
        one_hot_samples = tf.one_hot(samples, depth=self.local_hilbert_space, dtype=tf.float32)
        log_probs = 0.5 * tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs, one_hot_samples), axis=2)), axis=1)

        # # Logic in old version
        # data = tf.one_hot(samples[:,0:self.N-1],depth=self.K)
        # x0 = tf.zeros(shape=[num_samples,1,self.K],dtype=tf.float32)
        # inputs_old = tf.concat([x0,data],axis=1)
        # hidden_state = tf.zeros(shape=[num_samples,self.nh])
        # rnn_output,_ = self.rnn(inputs_old,initial_state = hidden_state)
        # probs_old    = self.dense(rnn_output)
        # log_probs   = 0.5 * tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.math.log(1e-10+probs),one_hot_samples),axis=2), axis=1)

        return log_probs



# Vectorized Energy Function, below no longer needed
#--------------------------------------------------------------------------------------------------------------------------------
    # #@tf.function
    # def localenergy(self,samples,logpsi):
    #     # print("local energy in model function!")
    #     eloc = tf.zeros(shape=[tf.shape(samples)[0]],dtype=tf.float32)
    #     # Chemical potential
    #     for j in range(self.N):
    #         eloc += - self.delta * tf.cast(samples[:,j],tf.float32)
    #     # print(eloc)
    #     # count=0
    #     for n in range(len(self.interactions)):
    #         contrib = (self.V/self.interactions[n][0]) * tf.cast(samples[:,self.interactions[n][1]]*samples[:,self.interactions[n][2]],tf.float32)
    #         # if np.all(contrib)>0:
    #         #     count+=1
    #         # print(contrib)
    #         eloc += contrib
    #     # print(count)
    #     # print(eloc)
    #     flip_logpsi = tf.zeros(shape=[tf.shape(samples)[0]])
    #     # Off-diagonal part
    #     for j in range(self.N):
    #         flip_samples = np.copy(samples)
    #         flip_samples[:,j] = 1 - flip_samples[:,j]
    #         flip_logpsi = self.logpsi(flip_samples)
    #         eloc += -0.5*self.Omega * tf.math.exp(flip_logpsi-logpsi)
    #     # print(eloc)
    #     return eloc

    # """ Generate the square lattice structures """
    # def coord_to_site(self,x,y):
    #     return self.Ly*x+y
    
    # def buildlattice(self):
    #     self.interactions = []
        
    #     for n in range(1,self.Lx):
    #         for n_ in range(n+1):
                
    #             if n+n_ > self.trunc:
    #                 continue
        
    #             else:
    #                 for x in range(self.Lx-n_):
    #                     for y in range(self.Ly-n):
    #                         coeff = np.sqrt(n**2+n_**2)**6
    #                         if n_ == 0 :
    #                             self.interactions.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x,y+n)])
    #                         elif n == n_: 
    #                             self.interactions.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x+n,y+n)])
    #                             self.interactions.append([coeff,self.coord_to_site(x+n,y),self.coord_to_site(x,y+n)])
    #                         else:
    #                             self.interactions.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x+n_,y+n)])
    #                             self.interactions.append([coeff,self.coord_to_site(x+n_,y),self.coord_to_site(x,y+n)])
                            
    #                 for y in range(self.Ly-n_):
    #                     for x in range(self.Lx-n):
    #                         coeff = np.sqrt(n**2+n_**2)**6
    #                         if n_ == 0 :
    #                             self.interactions.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x+n,y)])
    #                         elif n == n_: 
    #                             continue #already counted above
    #                         else:
    #                             self.interactions.append([coeff,self.coord_to_site(x,y),self.coord_to_site(x+n,y+n_)])
    #                             self.interactions.append([coeff,self.coord_to_site(x,y+n_),self.coord_to_site(x+n,y)])
