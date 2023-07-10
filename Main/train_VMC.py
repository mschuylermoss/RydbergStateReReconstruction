import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
from dset_helpers import load_exact_Es
from OneD_RNN import OneD_RNN_wavefxn, RNNWavefunction1D
from TwoD_RNN import MDRNNWavefunction,MDTensorizedRNNCell,MDRNNGRUcell
from energy_func import buildlattice,construct_mats,get_Rydberg_Energy_Vectorized
from stag_mag import calculate_stag_mag

def optimizer_initializer(optimizer):
    fake_var = tf.Variable(1.0)
    with tf.GradientTape() as tape:
        fake_loss = tf.reduce_sum(fake_var ** 2)
    grads = tape.gradient(fake_loss, [fake_var])
    # Ask the optimizer to apply the processed gradients.
    optimizer.apply_gradients(zip(grads, [fake_var]))

def Train_w_VMC(config):

    '''
    Run RNN using vmc sampling or qmc data. If qmc_data is None, uses vmc sampling. 
    Otherwise uses qmc data loaded in qmc_data
    '''

    # ---- System Parameters -----------------------------------------------------------------
    Lx = config['Lx']
    Ly = config['Ly']
    V = config['V']
    delta = config['delta']
    Omega = config['Omega']

    # ---- RNN Parameters ---------------------------------------------------------------------
    num_hidden = config['nh']
    learning_rate = config['lr']
    weight_sharing = config['weight_sharing']
    trunc = config['trunc']
    seed = config['seed']
    rnn_type = config['RNN']

    # ---- Training Parameters ----------------------------------------------------------------
    ns = config['ns']
    batch_samples = config.get('batch_samples', False)
    if batch_samples:
        batch_size_samples = config.get('batch_size_samples', 100)
        print(f"Batching samples drawn from RNN with batch size = {batch_size_samples}")
    else:
        batch_size_samples = ns
        print(f"Not batching samples drawn from RNN, meaning batch size = {ns}")
    epochs = config['VMC_epochs']
    global_step = tf.Variable(0, name="global_step")

    # ---- Data Path ---------------------------------------------------------------------------
    exp_name = config['name']
    path = f'./data/N_{Lx*Ly}/{exp_name}/{rnn_type}_rnn/delta_{delta}/seed_{seed}/VMC_only'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+'/config.txt', 'w') as file:
        for k,v in config.items():
            file.write(k+f'={v}\n')

    # ---- Initiate RNN Wave Function ----------------------------------------------------------
    if config['RNN'] == 'OneD':
        if config['Print'] ==True:
            print(f"Training a one-D RNN wave function with {num_hidden} hidden units and shared weights.")
        OneD_RNN_version = config.get('version', 'Old')
        if OneD_RNN_version=='New':
            wavefxn = RNNWavefunction1D(Lx,Ly,num_hidden,learning_rate,seed)
        else:
            wavefxn = OneD_RNN_wavefxn(Lx,Ly,num_hidden,learning_rate,seed)
    elif config['RNN'] =='TwoD':
        mdgru = config.get('MDGRU',True)
        if config['Print'] ==True:
            print(f"Training a two-D RNN wave function with {num_hidden} hidden units and shared weights = {weight_sharing}.")
        if mdgru:
            wavefxn = MDRNNWavefunction(Lx,Ly,V,Omega,delta,num_hidden,learning_rate,weight_sharing,trunc,seed,cell=MDRNNGRUcell)
        else:
            wavefxn = MDRNNWavefunction(Lx,Ly,V,Omega,delta,num_hidden,learning_rate,weight_sharing,trunc,seed,cell=MDTensorizedRNNCell)
    else:
        raise ValueError(f"{config['RNN']} is not a valid option for the RNN wave function. Please choose OneD or TwoD.")

    if config['Print'] ==True:
        print(f"The experimental parameters are: V = {V}, delta = {delta}, Omega = {Omega}.")
        print(f"The system is an array of {Lx} by {Ly} Rydberg Atoms.")

    # ---- Define Train Step --------------------------------------------------------------------
    interaction_list = buildlattice(Lx,Ly,trunc)
    O_mat,V_mat,coeffs = construct_mats(interaction_list, Lx*Ly)
    Ryd_Energy_Function = get_Rydberg_Energy_Vectorized(interaction_list,wavefxn.logpsi)
    Omega_tf = tf.constant(Omega)
    delta_tf = tf.constant(delta)
    V0_tf = tf.constant(V)

    @tf.function
    def train_step(training_samples):
        print("Tracing!")
        with tf.GradientTape() as tape:
            training_sample_logpsi = wavefxn.logpsi(training_samples)
            with tape.stop_recording():
                training_sample_eloc = Ryd_Energy_Function(Omega_tf,delta_tf,V0_tf,O_mat,V_mat,coeffs,training_samples,training_sample_logpsi)
                sample_Eo = tf.reduce_mean(training_sample_eloc)
            sample_loss = tf.reduce_mean(2.0*tf.multiply(training_sample_logpsi, tf.stop_gradient(training_sample_eloc)) - 2.0*tf.stop_gradient(sample_Eo)*training_sample_logpsi)
            gradients = tape.gradient(sample_loss, wavefxn.trainable_variables)
            wavefxn.optimizer.apply_gradients(zip(gradients, wavefxn.trainable_variables))
        return sample_loss

    # ---- Start From CKPT or Scratch -------------------------------------------------------------
    ckpt = tf.train.Checkpoint(step=global_step, optimizer=wavefxn.optimizer, variables=wavefxn.trainable_variables)
    manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=1)

    if config['CKPT']:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("CKPT ON and ckpt found.")
            print("Restored from {}".format(manager.latest_checkpoint))
            latest_ckpt = ckpt.step.numpy()
            optimizer_initializer(wavefxn.optimizer)
            print(f"Continuing at step {ckpt.step.numpy()}")
            energy = np.load(path+'/Energy.npy').tolist()[0:latest_ckpt]
            variance = np.load(path+'/Variance.npy').tolist()[0:latest_ckpt]
            stag_mag = np.load(path+'/StagMag.npy').tolist()[0:latest_ckpt]
            cost = np.load(path+'/Cost.npy').tolist()[0:latest_ckpt]

        else:
            print("CKPT ON but no ckpt found. Initializing from scratch.")
            energy = []
            variance = []
            stag_mag = []
            cost = []

    else:
        print("CKPT OFF. Initializing from scratch.")
        energy = []
        variance = []
        stag_mag = []
        cost = []

    # ---- Train ----------------------------------------------------------------------------------
    it = global_step.numpy()

    for n in range(it+1, epochs+1):
        samples, _ = wavefxn.sample(ns)
        samples_tf = tf.data.Dataset.from_tensor_slices(samples)
        samples_tf = samples_tf.batch(batch_size_samples)
        loss = []

        for i, batch in enumerate(samples_tf):
            batch_loss = train_step(batch)
            loss.append(batch_loss)

        global_step.assign_add(1)
        
        #append the energy to see convergence
        avg_loss = np.mean(loss)
        samples, _ = wavefxn.sample(ns)
        samples_tf = tf.data.Dataset.from_tensor_slices(samples)
        samples_tf = samples_tf.batch(batch_size_samples)
        energies = []
        for i, batch in enumerate(samples_tf):
            batch_logpsi = wavefxn.logpsi(batch)
            sample_eloc = Ryd_Energy_Function(Omega_tf,delta_tf,V0_tf,O_mat,V_mat,coeffs,batch,batch_logpsi)
            energies.append(sample_eloc.numpy())
        avg_E = np.mean(energies)/float(wavefxn.N)
        var_E = np.var(energies)/float(wavefxn.N)
        energy.append(avg_E)
        variance.append(var_E)
        _,_,avg_abs_stag_mag,_ = calculate_stag_mag(Lx,Ly,samples.numpy())
        stag_mag.append(avg_abs_stag_mag)
        cost.append(avg_loss)

        if (config['Print']) & (n%50 == 0):
            print(f"Step #{n}")
            print(f"Energy = {avg_E}")
            print(f"Variance = {var_E}")
            print(f"Staggered Magnetization = {avg_abs_stag_mag}")
            print(" ")

        if (config['CKPT']) & (n%50 == 0):
            manager.save()
            print(f"Saved checkpoint for step {n} in {path}.")

        if (config['Write_Data']) & (n%50 == 0):
            print(f"Saved training quantitites for step {n} in {path}.")
            np.save(path+'/Energy',energy)
            np.save(path+'/Variance',variance)
            np.save(path+'/StagMag',stag_mag)
            np.save(path+'/Cost',cost)
                
    return wavefxn, energy, variance, stag_mag, cost
