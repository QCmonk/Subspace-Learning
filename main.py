import os
import numpy as np 
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Layer, Dense

import qfunk.qoptic as qop
import qfunk.generator as qog
import qfunk.utility as qut
from utility import *

# bad josh
os.environ['KMP_DUPLICATE_LIB_OK']='True'

###############################################################
# whether to use advanced or random initialisation
advanced_mode = False
# network mode, is a pair of network <'single','deep'> 
network_mode = 'deep'
###############################################################

# ------------------------------------------------------------
# Section 0: Program parameter definitions
# ------------------------------------------------------------
# define optical settings
modes = 4
# photons
photons = 2
# number of epochs to train over
epochs = 10
# initial learning rate
init_lr = 1e-3
# batch size for learning
batch_size = 10
# number of training examples
data_num = 2000
# learning rate polynomial
lr_pow = 1
# number of layers for deep network
layer_num = 3
# whether or not to train the network or just compile it
train = True
# location and name of model save
modelfile_name = "testing_network_{}_{}".format(modes, photons)
# the seed to use for numpy random number generator
np.random.seed(1223334444)
# optical dimension
qdim = qop.fock_dim(modes, photons)
# ------------------------------------------------------------
# Section 1: Define basic network topology
# ------------------------------------------------------------
# should chuck this in a function call, bit too complex for main file 
if advanced_mode:

	# setup desired network
	if network_mode is 'single':
		# setup complete network
		input_state = Input(batch_shape=[None, qdim], dtype=tf.complex64, name="state_input")

		# setup optical network output
		dilated_layer = ULayer(modes=qdim, photons=1, vec=True, force=True) 
		output = dilated_layer(input_state)

		# define the model
		model = Model(inputs=input_state, outputs=output, name="Optical_Compute_Network")

		#------------------------------------------------

		# define input layer of network
		input_state_projected = Input(batch_shape=[None, modes], dtype=tf.complex64, name="state_input_simple")

		# setup optical network output
		projected_layer = ULayer(modes=modes, photons=1, vec=True, force=True)
		output_projected = projected_layer(input_state_projected)

		# define the model
		model_projected = Model(inputs=input_state_projected, outputs=output_projected, name="Projected_Compute_Network")

	if network_mode is 'deep':

		# setup complete network
		input_state = Input(batch_shape=[None, qdim], dtype=tf.complex64, name="state_input")

		# setup optical networks as callables
		dilated_layers = [ULayer(modes=qdim, photons=1, vec=True, force=True) for i in range(3)]

		# call dilated operator layers to construct network
		output = dilated_layers[0](input_state)
		output = tf.keras.activations.relu(output)
		# iterate over them
		for i in range(1,layer_num):
			output = dilated_layers[i](output)
			output = tf.keras.activations.relu(output)

		# final dense layer for categorization
		output = Dense(10)(output)

		# define the model
		model = Model(inputs=input_state, outputs=output, name="Optical_Compute_Network")

		#------------------------------------------------

		# define input layer of network
		input_state_projected = Input(batch_shape=[None, modes], dtype=tf.complex64, name="state_input_simple")

		# setup optical network output
		projected_layers = [ULayer(modes=modes, photons=1, vec=True, force=False) for i in range(3)]
		
		# call dilated operator layers to construct network
		output_projected = projected_layers[0](input_state_projected)
		output = tf.keras.activations.relu(output)
		# iterate over them
		for i in range(1,layer_num):
			output_projected = projected_layers[i](output_projected)
			output = tf.keras.activations.relu(output)

		# define the model
		model_projected = Model(inputs=input_state_projected, outputs=output_projected, name="Projected_Compute_Network")


else:
	if network_mode is 'single':
		# define input layer of network, no longer a time sequence to be considered
		input_state = Input(batch_shape=[None, qdim], dtype=tf.complex64, name="state_input")

		# setup optical network output
		output = ULayer(modes=qdim, photons=1, vec=True, force=True)(input_state)

		# define the model
		model = Model(inputs=input_state, outputs=output, name="Optical_Compute_Network")

	# setup network for a more significant learning task
	elif network_mode is 'deep':

		# define input layer of network, no longer a time sequence to be considered
		input_state = Input(batch_shape=[None, qdim], dtype=tf.complex64, name="state_input")


		# setup optical network
		output = ULayer(modes=qdim, photons=1, vec=True, force=True)(input_state)
		output = tf.keras.activations.relu(output)
		for i in range(layer_num-1):
			output = ULayer(modes=qdim, photons=1, vec=True, force=True)(output)
			output = tf.keras.activations.relu(output)
			
		# define the model
		model = Model(inputs=input_state, outputs=output, name="Optical_Compute_Network")


# ------------------------------------------------------------
# Section 2: Model compilation
# ------------------------------------------------------------

# define standard optimiser
opt = Adam(learning_rate=init_lr)

if network_mode is 'deep':
	# catergorical cross entropy for mnist
	tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
else:
	# define loss function
	loss = tf.keras.losses.MeanSquaredError()

# compile projected model if we made one 
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

if network_mode is 'single':
	# extract complete data
	problem_data = unitary_map_data_gen(qdim, data_num)

	# compute projected versions if running in advanced mode
	if advanced_mode:
		projected_problem_data = []
		for states in problem_data:
			projected_problem_data.append(data_projection(states, modes, photons))

		ptrain_input_states, ptrain_output_states, ptest_input_states, ptest_output_states = projected_problem_data


	# extract partitioned data
	train_inputs, train_outputs, test_inputs, test_outputs = problem_data

elif network_mode is 'deep':

	# extract complete data
	train_inputs, train_outputs, test_inputs, test_outputs = mnist_data_gen(qdim, data_num)

	# compute projected versions if running in advanced mode
	if advanced_mode:
		ptrain_inputs = data_projection(train_input_states, modes, photons)
		ptrain_outputs = 
		ptest_inputs
		ptest_outputs 

		projected_problem_data.append(data_projection(states, modes, photons))

		ptrain_input_states, ptrain_output_states, ptest_input_states, ptest_output_states 



# ------------------------------------------------------------
# Section 4: Model training
# ------------------------------------------------------------

if train:
	# train simplified model
	if advanced_mode:

		if network_mode is 'single':
			model_projected.fit(x=ptrain_inputs,
					            y=ptrain_outputs,
					            epochs=50,
					            steps_per_epoch=len(ptrain_inputs)//batch_size,
					            verbose=1,
					            validation_data=(ptest_inputs, ptest_outputs),
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
		              steps_per_epoch=len(train_inputs)//batch_size,
		              verbose=1,
		              validation_data=(test_inputs, test_outputs),
		              validation_steps=1,
		              callbacks=[PerformanceCallBack(name='advanced')])


		elif network_mode is 'deep':

			# train the simplified network
			model_projected.fit(x=ptrain_inputs,
					            y=ptrain_outputs,
					            epochs=30,
					            steps_per_epoch=len(ptrain_inputs)//batch_size,
					            verbose=1,
					            validation_data=(ptest_inputs, ptest_outputs),
					            validation_steps=1,
					            callbacks=[]),
			
			# generate symmetric map
			S = qop.symmetric_map(modes, photons)

			# now iterate over each layer
			for j,unitary_layer in enumerate(projected_layers):
				# extract trained unitary parameters
				diag = unitary_layer.diag.numpy()
				theta = unitary_layer.theta.numpy()
				phi = unitary_layer.phi.numpy()
				bms_spec = unitary_layer.bms_spec

				# build complete unitary explicitly using same stich method
				U = tf_clements_stitch(bms_spec, theta, phi, diag).numpy()
				# dilate to complete space
				Udilate = S @ qut.mkron(U, photons) @ qut.dagger(S)

				# compute unitary parameters
				tlist, new_diag = clements_phase_end(Udilate, tol=1e-7)

				# now assign these dilated values to this initial values of the complete unitary map
				new_theta = np.zeros_like(dilated_layers[j].theta.numpy())
				new_phi = np.zeros_like(dilated_layers[j].phi.numpy())
				for i,info in enumerate(tlist):
					new_theta[i] = info[2]
					new_phi[i] = info[3]

				# assign initialisation values
				dilated_layers[j].theta.assign(new_theta)
				dilated_layers[j].phi.assign(new_phi)
				dilated_layers[j].diag.assign(np.angle(new_diag))


			# train full model
			model.fit(x=train_input_states,
		              y=train_output_states,
		              epochs=epochs,
		              steps_per_epoch=len(train_inputs)//batch_size,
		              verbose=1,
		              validation_data=(test_inputs, test_outputs),
		              validation_steps=1,
		              callbacks=[PerformanceCallBack(name='advanced')])


	else:
		model.fit(x=train_input_states,
	              y=train_output_states,
	              epochs=epochs,
	              steps_per_epoch=len(train_inputs)//batch_size,
	              verbose=1,
	              validation_data=(test_inputs, test_outputs),
	              validation_steps=1,
	              callbacks=[PerformanceCallBack(name='simple')])#model_checkpoint_callback,

