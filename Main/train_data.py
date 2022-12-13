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

def Train_w_Data(config):

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
    batch_size = config['batch_size']
    epochs = config['Data_epochs']
    data = data_given_param(sweep_rate,delta)
    tf_dataset = create_KZ_tf_dataset(data)   
    global_step = tf.Variable(0, name="global_step") 

    # ---- Data Path ---------------------------------------------------------------------------
    exp_name = config['name']
    path = f'./data/N_{Lx*Ly}/delta_{delta}/{rnn_type}_rnn/{exp_name}'
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
        if config['Print'] ==True:
            print(f"Training a two-D RNN wave function with {num_hidden} hidden units and shared weights = {weight_sharing}.")
        if config['MDGRU']:
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
    def train_step(input_batch):
        print("Tracing!")
        with tf.GradientTape() as tape:
            logpsi = wavefxn.logpsi(input_batch)
            loss = - 2.0 * tf.reduce_mean(logpsi)
        # Compute the gradients either with qmc_loss
        gradients = tape.gradient(loss, wavefxn.trainable_variables)
        clipped_gradients = [tf.clip_by_value(g, -10., 10.) for g in gradients]
        # Update the parameters
        wavefxn.optimizer.apply_gradients(zip(clipped_gradients, wavefxn.trainable_variables))
        return loss

    # ---- Start From CKPT or Scratch -------------------------------------------------------------
    ckpt = tf.train.Checkpoint(step=global_step, optimizer=wavefxn.optimizer, variables=wavefxn.trainable_variables)
    manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=1)

    if config['CKPT']:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("CKPT ON and ckpt found.")
            print("Restored from {}".format(manager.latest_checkpoint))
            latest_ckpt = ckpt.step.numpy()
            start_from_ckpt = True
            optimizer_initializer(wavefxn.optimizer)
            print(f"Continuing at step {ckpt.step.numpy()}")
            energy = np.load(path+'/Energy.npy').tolist()[0:latest_ckpt]
            variance = np.load(path+'/Variance.npy').tolist()[0:latest_ckpt]
            cost = np.load(path+'/Cost.npy').tolist()[0:latest_ckpt]

        else:
            print("CKPT ON but no ckpt found. Initializing from scratch.")
            latest_ckpt = 0
            start_from_ckpt = False
            energy = []
            variance = []
            cost = []

    else:
        print("CKPT OFF. Initializing from scratch.")
        start_from_ckpt = False
        latest_ckpt = 0
        energy = []
        variance = []
        cost = []

    # ---- Train ----------------------------------------------------------------------------------
    it = global_step.numpy()

    for n in range(it+1, epochs+1):
        #use data to update RNN weights
        dset = tf_dataset.shuffle(len(tf_dataset))
        dset = dset.batch(batch_size)
        loss = []

        for i, batch in enumerate(dset):
            # Evaluate the loss function in AD mode
            batch_loss = train_step(batch)
            loss.append(batch_loss)

        global_step.assign_add(1)

        #append the energy to see convergence
        avg_loss = np.mean(loss)
        samples, _ = wavefxn.sample(ns)
        sample_logpsi = wavefxn.logpsi(samples)
        sample_eloc = Ryd_Energy_Function(Omega_tf,delta_tf,V0_tf,O_mat,V_mat,coeffs,samples,sample_logpsi)
        energies = sample_eloc.numpy()
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

        if (config['CKPT']) & (n%10 == 0): # checkpoint frequently during data training
            manager.save()
            print(f"Saved checkpoint for step {n} in {path}.")
        
        if (config['Write_Data']) & (n%10 == 0): # need to save training quantities each time we checkpoint
            print(f"Saved training quantitites for step {n} in {path}.")
            np.save(path+'/Energy',energy)
            np.save(path+'/Variance',variance)
            np.save(path+'/Cost',cost)
            np.save(path+'/Samples',[])
    
    # ---- Final Save -----------------------------------------------------------------------------------
    if config['Write_Data']:
        print(f"Saved training quantities for the FINAL step {n} in {path}.")
        samples_final,_ = wavefxn.sample(10000)
        np.save(path+'/Energy',energy)
        np.save(path+'/Variance',variance)
        np.save(path+'/Cost',cost)
        np.save(path+'/Samples',samples_final)

    return wavefxn, energy, variance,cost
