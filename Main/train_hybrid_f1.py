import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
from dset_helpers import create_KZ_tf_dataset, data_given_param
from OneD_RNN import OneD_RNN_wavefxn,RNNWavefunction1D
from TwoD_RNN import MDRNNWavefunction,MDTensorizedRNNCell,MDRNNGRUcell
from energy_func import buildlattice,construct_mats,get_Rydberg_Energy_Vectorized

def optimizer_initializer(optimizer):
    fake_var = tf.Variable(1.0)
    with tf.GradientTape() as tape:
        fake_loss = tf.reduce_sum(fake_var ** 2)
    grads = tape.gradient(fake_loss, [fake_var])
    # Ask the optimizer to apply the processed gradients.
    optimizer.apply_gradients(zip(grads, [fake_var]))

def Train_w_Data_then_VMC(config):

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
    sweep_rate = config['sweep_rate']

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
        if config['Print'] ==True:
            print(f"Batching samples drawn from RNN with batch size = {batch_size_samples}")
    else:
        batch_size_samples = ns
        if config['Print'] ==True:
            print(f"Not batching samples drawn from RNN, meaning batch size = {ns}")
    batch_size_data = config.get('batch_size_data', 100)
    data_epochs = config['Data_epochs']
    qmc_data = config.get('QMC_data', False)
    vmc_epochs = config['VMC_epochs']
    total_epochs = data_epochs+vmc_epochs
    global_step = tf.Variable(0, name="global_step") 

    # ---- Data Path ---------------------------------------------------------------------------
    exp_name = config['name']
    # Add-ons for Figure 1
    if qmc_data:
        dset_size = config.get('dset_size',1000)
        path_old = f'./data/N_{Lx*Ly}/{exp_name}/QMC_data/dset_size_{dset_size}/nh_{num_hidden}/{rnn_type}_rnn/delta_{delta}/seed_{seed}'
        hybrid_addition = f'/hybrid_training/lr_{learning_rate}/{data_epochs}_datasteps'
        path_new = path_old+hybrid_addition
    else:
        path_old = f'./data/N_{Lx*Ly}/{exp_name}/Exp_data/nh_{num_hidden}/{rnn_type}_rnn/delta_{delta}/seed_{seed}'
        hybrid_addition = f'/hybrid_training/lr_{learning_rate}/{data_epochs}_datasteps'
        path_new = path_old+hybrid_addition
    # path_old = f'./data/N_{Lx*Ly}/{exp_name}/{rnn_type}_rnn/delta_{delta}/seed_{seed}'
    # path_new = f'./data/N_{Lx*Ly}/{exp_name}/{rnn_type}_rnn/delta_{delta}/seed_{seed}/hybrid_training/{data_epochs}_datasteps'
    # path_new = f'./data/N_{Lx*Ly}/{exp_name}/{rnn_type}_rnn/delta_{delta}/seed_{seed}/hybrid_training/lr_{learning_rate}/{data_epochs}_datasteps'
    if not os.path.exists(path_new):
        os.makedirs(path_new)
    with open(path_new+'/config.txt', 'w') as file:
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
            print("Using GRU cell.")
            wavefxn = MDRNNWavefunction(Lx,Ly,V,Omega,delta,num_hidden,learning_rate,weight_sharing,trunc,seed,cell=MDRNNGRUcell)
        else:
            wavefxn = MDRNNWavefunction(Lx,Ly,V,Omega,delta,num_hidden,learning_rate,weight_sharing,trunc,seed,cell=MDTensorizedRNNCell)
    else:
        raise ValueError(f"{config['RNN']} is not a valid option for the RNN wave function. Please choose OneD or TwoD.")

    if config['Print'] ==True:
        print(f"The experimental parameters are: V = {V}, delta = {delta}, Omega = {Omega}.")
        print(f"The system is an array of {Lx} by {Ly} Rydberg Atoms")

    # ---- Define Train Step --------------------------------------------------------------------
    interaction_list = buildlattice(Lx,Ly,trunc)
    O_mat,V_mat,coeffs = construct_mats(interaction_list, Lx*Ly)
    Ryd_Energy_Function = get_Rydberg_Energy_Vectorized(interaction_list,wavefxn.logpsi)
    Omega_tf = tf.constant(Omega)
    delta_tf = tf.constant(delta)
    V0_tf = tf.constant(V)
    
    @tf.function
    def train_step_VMC(training_samples):
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
    ckpt = tf.train.Checkpoint(step=global_step, optimizer=wavefxn.optimizer, variables=wavefxn.trainable_variables) # these are trackable objects - we get to specify
    manager_old = tf.train.CheckpointManager(ckpt, path_old, max_to_keep=1)
    manager_new = tf.train.CheckpointManager(ckpt, path_new, max_to_keep=1)
    load_only = config.get('Load_Only', False)

    if load_only:
        print(f"Restoring the fully trained model from {manager_new.latest_checkpoint}.")
        manager_new.latest_checkpoint
        ckpt_step = ckpt.step.numpy()
        optimizer_initializer(wavefxn.optimizer)
        energy = np.load(path_new+'/Energy.npy').tolist()[0:ckpt_step]
        variance = np.load(path_new+'/Variance.npy').tolist()[0:ckpt_step]
        cost = np.load(path_new+'/Cost.npy').tolist()[0:ckpt_step]
        global_step = tf.constant(total_epochs) # so that we dont keep training
    else:
        if config['CKPT']: # will start from a specified starting point OR from the latest checkpoint if specified checkpoint not found OR from scratch if neither are found
            restart_point = data_epochs
            index = int((restart_point-(restart_point%10))/10 - 1)
            if index < len(manager_old.checkpoints):
                ckpt.restore(manager_old.checkpoints[index])
                ckpt_found = True
            else:
                manager_old.latest_checkpoint
                ckpt_found = False
            if ckpt_found:
                print(f"CKPT ON and ckpt {index} found.")
                print("Restored from {}".format(manager_old.checkpoints[index]))
                ckpt_step = ckpt.step.numpy()
                optimizer_initializer(wavefxn.optimizer)
                print(f"Continuing at step {ckpt.step.numpy()}")
                energy = np.load(path_old+'/Energy.npy').tolist()[0:ckpt_step]
                print("ENERGY LEN AFTER CKPT: ",len(energy))
                variance = np.load(path_old+'/Variance.npy').tolist()[0:ckpt_step]
                cost = np.load(path_old+'/Cost.npy').tolist()[0:ckpt_step]
                wavefxn.optimizer.lr = learning_rate
            elif manager_old.latest_checkpoint:
                print(f"CKPT ON but ckpt {index} not found.")
                print("Restored from {}".format(manager_old.latest_checkpoint))
                latest_ckpt = ckpt.step.numpy()
                optimizer_initializer(wavefxn.optimizer)
                print(f"Continuing at step {ckpt.step.numpy()}")
                energy = np.load(path_old+'/Energy.npy').tolist()[0:latest_ckpt]
                variance = np.load(path_old+'/Variance.npy').tolist()[0:latest_ckpt]
                cost = np.load(path_old+'/Cost.npy').tolist()[0:latest_ckpt]
                wavefxn.optimizer.lr = learning_rate
            else:
                print("CKPT ON but no ckpt found. Initializing from scratch.")
                latest_ckpt = 0
                energy = []
                variance = []
                cost = []
        else:
            print("CKPT OFF. Initializing from scratch.")
            latest_ckpt = 0
            energy = []
            variance = []
            cost = []

    # ---- Train ----------------------------------------------------------------------------------
    it = global_step.numpy()

    for n in range(it+1, total_epochs+1):
        samples, _ = wavefxn.sample(ns)
        samples_tf = tf.data.Dataset.from_tensor_slices(samples)
        samples_tf = samples_tf.batch(batch_size_samples)
        loss = []

        for i, batch in enumerate(samples_tf):
            batch_loss = train_step_VMC(batch)
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
        cost.append(avg_loss)

        if (config['Print']) & (n%50 == 0):
            print(f"Step #{n}")
            print(f"Energy = {avg_E}")
            print(f"Variance = {var_E}")
            print(" ")

        if (config['CKPT']) & (n%50 == 0):
            manager_new.save()
            print(f"Saved checkpoint for step {n} in {path_new}.")

        if (config['Write_Data']) & (n%50 == 0):
            print(f"Saved training quantitites for step {n} in {path_new}.")
            np.save(path_new+'/Energy',energy)
            np.save(path_new+'/Variance',variance)
            np.save(path_new+'/Cost',cost)

    return wavefxn, energy, variance,cost
