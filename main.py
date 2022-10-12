import os
import numpy as np 
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Layer

import qfunk.qoptic as qop
import qfunk.generator as qog
import qfunk.utility as qut
from utility import *

# bad josh
os.environ['KMP_DUPLICATE_LIB_OK']='True'

###############################################################
# whether to use advanced or random initialisation
advanced_mode = False
# network mode, is a pair of network <'single','deep'> and layer type <'linear', 'unitary'>
network_mode = ['single', 'unitary']
###############################################################

# ------------------------------------------------------------
# Section 0: Program parameter definitions
# ------------------------------------------------------------
# define optical settings
modes = 5
# photons
photons = 3
# number of epochs to train over
epochs = 100
# initial learning rate
init_lr = 1e-2
# batch size for learning
batch_size = 10
# number of training examples
data_num = 1000
# learning rate polynomial
lr_pow = 1
# whether or not to train the network or just compile it
train = True
# whether to save model
save = True
# location and name of model save
modelfile_name = "testing_network_{}_{}".format(modes, photons)
# the seed to use for numpy random number generator
np.random.seed(1223334444)
# optical dimension
qdim = qop.fock_dim(modes, photons)
# ------------------------------------------------------------
# Section 1: Define basic network topology
# ------------------------------------------------------------

# setup advanced training mode if requested 
if advanced_mode:
	# setup complete network
	input_state = Input(batch_shape=[None, qdim], dtype=tf.complex64, name="state_input")

	# setup optical network output
	dilated_layer = ULayer(modes=qdim, photons=1, vec=True, force=True) 
	output = dilated_layer(input_state)

	# define the model
	model = Model(inputs=input_state, outputs=output, name="Optical_Compute_Network")


	# setup simple network
	# define input layer of network
	input_state_projected = Input(batch_shape=[None, modes], dtype=tf.complex64, name="state_input_simple")

	# setup optical network output
	projected_layer = ULayer(modes=modes, photons=1, vec=True, force=True)
	output_projected = projected_layer(input_state_projected)

	# define the model
	model_projected = Model(inputs=input_state_projected, outputs=output_projected, name="Projected_Compute_Network")


else:
	# define input layer of network, no longer a time sequence to be considered
	input_state = Input(batch_shape=[None, qdim], dtype=tf.complex64, name="state_input")

	# setup optical network output
	output = ULayer(modes=qdim, photons=1, vec=True, force=True)(input_state)

	# define the model
	model = Model(inputs=input_state, outputs=output, name="Optical_Compute_Network")



# define standard optimiser
opt = Adam(learning_rate=init_lr)

# define loss function
loss = tf.keras.losses.MeanSquaredError()
# ------------------------------------------------------------
# Section 2: Model compilation
# ------------------------------------------------------------
if advanced_mode:
	# compile it 
	model_projected.compile(optimizer=opt,
							  loss=loss,
							  metrics=[tf.keras.metrics.CosineSimilarity()])

# compile it
model.compile(optimizer=opt,
			  loss=loss,
			  metrics=[tf.keras.metrics.CosineSimilarity()])

# setup callbacks
name = datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = "C:\\Users\\Joshua\\Projects\\Research\\Archive\\Lightmind\\Logs\\fit\\"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir+name, write_graph=True)


model.summary()
# ------------------------------------------------------------
# Section 3: Data generation
# ------------------------------------------------------------

# extract complete data
problem_data = problem_data_gen(qdim, data_num)

# compute projected versions if running in advanced mode
if advanced_mode:
	projected_problem_data = []
	for states in problem_data:
		projected_problem_data.append(data_projection(states, modes, photons))

	ptrain_input_states, ptrain_output_states, ptest_input_states, ptest_output_states = projected_problem_data


# extract partitioned data
train_input_states, train_output_states, test_input_states, test_output_states = problem_data
# ------------------------------------------------------------
# Section 4: Model training
# ------------------------------------------------------------

if advanced_mode:


	# train simplified model
	if train:
		model_projected.fit(x=ptrain_input_states,
				            y=ptrain_output_states,
				            epochs=20,
				            steps_per_epoch=len(ptrain_input_states)//batch_size,
				            verbose=1,
				            validation_data=(ptest_input_states, ptest_output_states),
				            validation_steps=1,
				            callbacks=[]),
		
		# extract trained unitary parameters
		diag = projected_layer.diag.numpy()
		theta = projected_layer.theta.numpy()
		phi = projected_layer.phi.numpy()
		bms_spec = projected_layer.bms_spec

		# build complete unitary explicitly using same stich method
		U = tf_clements_stitch(bms_spec, theta, phi, diag).numpy()

		# dilate to complete space
		S = qop.symmetric_map(modes, photons)
		Udilate = S @ qut.mkron(U, photons) @ qut.dagger(S)

		# compute unitary parameters
		tlist, new_diag = clements_phase_end(Udilate, tol=1e-7)

		# now assign these dilated values to this initial values of the complete unitary map
		new_theta = np.zeros_like(dilated_layer.theta.numpy())
		new_phi = np.zeros_like(dilated_layer.phi.numpy())
		for i,info in enumerate(tlist):
			new_theta[i] = info[2]
			new_phi[i] = info[3]

		# assign initialisation values
		dilated_layer.theta.assign(new_theta)
		dilated_layer.phi.assign(new_phi)
		dilated_layer.diag.assign(np.angle(new_diag))

		# train simplified model
		model.fit(x=train_input_states,
	              y=train_output_states,
	              epochs=epochs,
	              steps_per_epoch=len(train_input_states)//batch_size,
	              verbose=1,
	              validation_data=(test_input_states, test_output_states),
	              validation_steps=1,
	              callbacks=[PerformanceCallBack(name='advanced')])




else:
	# train  model
	if train:
		model.fit(x=train_input_states,
	              y=train_output_states,
	              epochs=epochs,
	              steps_per_epoch=len(train_input_states)//batch_size,
	              verbose=1,
	              validation_data=(test_input_states, test_output_states),
	              validation_steps=1,
	              callbacks=[PerformanceCallBack(name='simple')])#model_checkpoint_callback,










# train full model using the initial set
