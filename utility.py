from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import qfunk.qoptic as qop
import qfunk.generator as qog
import qfunk.utility as qut



def problem_data_gen(dim, num, partition=0.9):
    """
    Generates problem data for a complete unitary operator
    """

    # generate a random unitary map
    U = qog.random_unitary(dim)

    # generate a bunch of random quantum input states
    input_states = np.random.randn(num, dim) + 1j*np.random.randn(num, dim)

    # normalise all of these
    norms = 1/np.linalg.norm(input_states, ord=2, axis=1)   

    # apply normalisation
    input_states = np.einsum('i,ik->ik', norms, input_states)

    # now apply unitary to get output states
    output_states = np.einsum('ij, kj->ki', U, input_states)

    # partition data
    partition_ind = int(partition*num)
    train_input_states = input_states[:partition_ind,:]
    train_output_states = output_states[:partition_ind,:]
    test_input_states = input_states[partition_ind:,:]
    test_output_states = output_states[partition_ind:,:]

    return train_input_states, train_output_states, test_input_states, test_output_states



def data_projection(A, modes, photons):
    """
    Projects input data down to a much reduced subspace 
    """ 
    # construct data projection into this space
    dim = qop.fock_dim(modes, photons)
    pdim = modes

    # compute overhead number states
    multi_states = qop.number_states(modes, photons)
    # compute local number states
    single_states = qop.number_states(modes, 1)
    # generate lookup gen
    multi_lookup = qop.lookup_gen(modes, photons)
    single_lookup = qop.lookup_gen(modes, 1)

    # basis sets
    multi_basis = np.eye(dim)
    single_basis = np.eye(pdim)

    # iterate over all elements of the input space
    M = 0.0
    for state in multi_states:
        # first generate the input strings for lookup
        state_str = ''.join(str(s) for s in state)
    
        # get index and find relevant basis element
        index = multi_lookup[state_str]
        input_vec = multi_basis[:,index].reshape([-1,1])


        # now generate the substates it is mapped to
        cnt = 0
        K = 0.0
        for pos,l in enumerate(state):
            # check if mode is occupied by anything
            if l>0:
                cnt+=1
                # generate single photon string
                state_str_proj = '0'*(pos) + '1' + '0'*(modes-pos-1)
                # get basis element
                index = single_lookup[state_str_proj]
                # add to output state TODO: This is a problem: (|i>+|j>+|k>)<m| A |n>(<a|+<b|+<c|)
                substate = single_basis[:,index].reshape([-1,1])


                K += np.kron(substate, qut.dagger(input_vec))

        # normalise the output 
        K /= np.sqrt(cnt)

        M += K

    # hit the data with the projection operator
    new_states = np.einsum('ij, kj->ki', M, A)

    return new_states


def tf_clements_stitch(beam_spec, theta, phi, diag, rev=tf.constant(True)):
    """
    Computes the unitary given by a clements decomposition with tensorflow compliance
    """

    nmax = beam_spec[0][-1]

    # construct adjoint of the phase shifts
    U = tf.linalg.diag(tf.exp(tf.complex(0.0, diag)))

    # iterate over specified beam splitter arrays
    for i, spec in enumerate(beam_spec):
        # construct nd scatter indices
        m, n = spec[:2]

        # construct indices to update with beam splitter params
        indices = index_gen(m,n,nmax)

        # retrieve iterations variable set
        th = tf.slice(theta, tf.constant([i]), size=tf.constant([1]))
        ph = tf.slice(phi, tf.constant([i]), size=tf.constant([1]))

        # construct beam splitter entries with compliant datatypes
        a = tf.math.exp(tf.complex(0.0, ph))*tf.complex(tf.cos(th), 0.0)
        b = tf.complex(-tf.sin(th),0.0)
        c = tf.math.exp(tf.complex(0.0, ph))*tf.complex(tf.sin(th), 0.0)
        d = tf.complex(tf.cos(th), 0.0)


        # concatenate matrix elements into vector for scatter operation
        var_list = [tf.constant([1.0+0.0j], dtype=tf.complex64,shape=[1,])]*(2+nmax)
        var_list[:4] = [a, b, c, d]
        var_list = tf.stack(var_list, 0, name="varlist_{}".format(i))
        # place variables with appropriate functionals (see Clements paper for ij=mn variable maps)
        Tmn = tf.scatter_nd(indices, var_list, tf.constant([nmax**2,1], dtype=tf.int64))
        # reshape into rank 2 tensor
        
        # cannot use update as the variable reference does not seem to like gradient computation
        #Tmn = tf.scatter_nd_add(tf.zeros((nmax**2,), dtype=tf.complex64), indices, var_list, name="scatter_{}".format(i))
        Tmn = tf.reshape(Tmn, tf.constant([nmax]*2))

        # return unitary using specified order convention
        U = tf.cond(rev, lambda: tf.matmul(U, Tmn), lambda: tf.matmul(Tmn, U))

    return U


def T(m, n, theta, phi, nmax):
    r"""The Clements T matrix from Eq. 1 of Clements et al. (2016)"""
    mat = np.identity(nmax, dtype=np.complex64)
    mat[m, m] = np.exp(1j*phi)*np.cos(theta)
    mat[m, n] = -np.sin(theta)
    mat[n, m] = np.exp(1j*phi)*np.sin(theta)
    mat[n, n] = np.cos(theta)

    return mat


def Ti(m, n, theta, phi, nmax):
    r"""The inverse Clements T matrix"""
    return np.transpose(T(m, n, theta, -phi, nmax))


def nullTi(m, n, U):
    r"""Nullifies element m,n of U using Ti"""
    (nmax, mmax) = U.shape

    if nmax != mmax:
        raise ValueError("U must be a square matrix")

    if U[m, n+1] == 0:
        thetar = np.pi/2
        phir = 0
    else:
        r = U[m, n] / U[m, n+1]
        thetar = np.arctan(np.abs(r))
        phir = np.angle(r)

    return [n, n+1, thetar, phir, nmax]


def nullT(n, m, U):
    r"""Nullifies element n,m of U using T"""
    (nmax, mmax) = U.shape

    if nmax != mmax:
        raise ValueError("U must be a square matrix")

    if U[n-1, m] == 0:
        thetar = np.pi/2
        phir = 0
    else:
        r = -U[n, m] / U[n-1, m]
        thetar = np.arctan(np.abs(r))
        phir = np.angle(r)

    return [n-1, n, thetar, phir, nmax]


def index_gen(m,n,nmax):
    """
    Generates index pair mappings for scatter update
    """
    rows,cols= [m, m, n, n], [m, n, m, n]
    for i in range(nmax):
        # skip values that will be covered by beam splitter
        if (i == m) or (i==n): continue

        rows.append(i)
        cols.append(i)

    # comute index mappings when reshaped to rank one tensor
    indices = np.ravel_multi_index([rows, cols], (nmax, nmax)).reshape(nmax+2, 1).tolist()
    return tf.constant(indices, dtype=tf.int64)



def clements(V, tol=1e-11):
    r"""Clements decomposition of a unitary matrix, with local
    phase shifts applied between two interferometers.
    See :ref:`clements` or :cite:`clements2016` for more details.
    This function returns a circuit corresponding to an intermediate step in
    Clements decomposition as described in Eq. 4 of the article. In this form,
    the circuit comprises some T matrices (as in Eq. 1), then phases on all modes,
    and more T matrices.
    The procedure to construct these matrices is detailed in the supplementary
    material of the article.
    Args:
        V (array[complex]): unitary matrix of size n_size
        tol (float): the tolerance used when checking if the matrix is unitary:
            :math:`|VV^\dagger-I| \leq` tol
    Returns:
        tuple[array]: tuple of the form ``(tilist,tlist,np.diag(localV))``
            where:
            * ``tilist``: list containing ``[n,m,theta,phi,n_size]`` of the Ti unitaries needed
            * ``tlist``: list containing ``[n,m,theta,phi,n_size]`` of the T unitaries needed
            * ``localV``: Diagonal unitary sitting sandwiched by Ti's and the T's
    """
    localV = V
    (nsize, _) = localV.shape

    # diffn = np.linalg.norm(V @ V.conj().T - np.identity(nsize))
    # if diffn >= tol:
    #     raise ValueError("The input matrix is not unitary")

    tilist = []
    tlist = []
    for k, i in enumerate(range(nsize-2, -1, -1)):
        if k % 2 == 0:
            for j in reversed(range(nsize-1-i)):
                tilist.append(nullTi(i+j+1, j, localV))
                localV = localV @ Ti(*tilist[-1])
        else:
            for j in range(nsize-1-i):
                tlist.append(nullT(i+j+1, j, localV))
                localV = T(*tlist[-1]) @ localV

    return tilist, tlist, np.diag(localV)

def clements_phase_end(V, tol=1e-11):
    r"""Clements decomposition of a unitary matrix.
    See :cite:`clements2016` for more details.
    Final step in the decomposition of a given discrete unitary matrix.

    Args:
        V (array[complex]): unitary matrix of size n_size
        tol (float): the tolerance used when checking if the matrix is unitary:
            :math:`|VV^\dagger-I| \leq` tol
    Returns:
        tuple[array]: returns a tuple of the form ``(tlist,np.diag(localV))``
            where:
            * ``tlist``: list containing ``[n,m,theta,phi,n_size]`` of the T unitaries needed
            * ``localV``: Diagonal unitary matrix to be applied at the end of circuit
    """
    tilist, tlist, diags = clements(V, tol)
    new_tlist, new_diags = tilist.copy(), diags.copy()

    # Push each beamsplitter through the diagonal unitary
    for i in reversed(tlist):
        em, en = int(i[0]), int(i[1])
        alpha, beta = np.angle(new_diags[em]), np.angle(new_diags[en])
        theta, phi = i[2], i[3]

        # The new parameters required for D',T' st. T^(-1)D = D'T'
        new_theta = theta
        new_phi = np.fmod((alpha - beta + np.pi), 2*np.pi)
        new_alpha = beta - phi + np.pi
        new_beta = beta

        new_i = [i[0], i[1], new_theta, new_phi, i[4]]
        new_diags[em], new_diags[en] = np.exp(
            1j*new_alpha), np.exp(1j*new_beta)

        new_tlist = new_tlist + [new_i]

    return (new_tlist, new_diags)



class ULayer(Layer):
    """
    Subclass Keras because I'm a busy guy. Untitary layer using Clements 2016 decomposition, universal 
    for single photon input. 
    """
    # initialise as subclass of keras general layer description
    def __init__(self, modes, photons=1, dim=None, u_noise=None, pad=0, vec=False, full=False, force=False, **kwargs):
        # layer identifier
        self.id = "unitary_layer"
        # pad dimensions
        self.pad = pad 
        # number of modes and number of pad modes
        self.modes = modes + self.pad
        # number of variables in operator
        self.vars = (self.modes**2 - self.modes)//2
        # number of photons
        self.photons = photons
        # pad modes - becomes dimension with 1 photon which we generally desire for universality
        self.pad = pad
        # whether to expect vector inputs
        self.vec = tf.constant(vec)
        # emergent dimension
        self.input_dim = self.output_dim = qop.fock_dim(modes, photons) + self.pad
        # catch extreme cases
        if self.input_dim > 100 and not force:
            raise ValueError(
                "System dimension is large ({}), decrease system dimension or set force flag to True".format(self.input_dim))
        
        # keep local copy of beam splitter decomposition table
        # TODO: This is such a cop out
        self.bms_spec = clements_phase_end(np.eye(self.modes))[0]

        # pass additional keywords to superclass initialisation
        super(ULayer, self).__init__(**kwargs)


    def build(self, input_shape):
        """Creates the variables of the layer (optional, for subclass implementers).

        This is a method that implements of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.

        This is typically used to create the weights of `Layer` subclasses.

        Arguments:
          input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).
        """

        # define weight initialisers with very tight distribution - corresponds to an identity
        with tf.init_scope():
            diag_init = tf.initializers.RandomNormal(mean=0, stddev=np.pi)
            theta_init = tf.initializers.RandomNormal(mean=np.pi, stddev=np.pi)
            phi_init = tf.initializers.RandomNormal(mean=-np.pi, stddev=np.pi)

            # superclass method for variable construction, get_variable has trouble with model awareness
            self.diag = self.add_weight(name="diags",
                                        shape=[self.modes],
                                        dtype=tf.float32,
                                        initializer=diag_init,
                                        trainable=True)

            self.theta = self.add_weight(name='theta',
                                         shape=[self.vars],
                                         dtype=tf.float32,
                                         initializer=theta_init,
                                         trainable=True)

            self.phi = self.add_weight(name='phi',
                                       shape=[self.vars],
                                       dtype=tf.float32,
                                       initializer=phi_init,
                                       trainable=True)


        # construct single photon unitary
        self.unitary = tf_clements_stitch(self.bms_spec, self.theta, self.phi, self.diag)

        # # construct multiphoton unitary using memory hungry (but uber fast) method
        # if self.photons > 1:
        #     if self.full:
        #         # preallocate on zero dimensional subspace
        #         U = tf.linalg.LinearOperatorFullMatrix(tf.constant([[1.0]], dtype=tf.complex64))

        #         for pnum in range(1,self.photons+1):
        #             # use symmetric map to compute multi photon unitary
        #             S = tf.constant(qop.symmetric_map(
        #             self.modes, pnum), dtype=tf.complex64)
        #             # map to product state then use symmetric isometry to reduce to isomorphic subspace
        #             V = tf.matmul(S, tf.matmul(tf_multikron(self.unitary, pnum), tf.linalg.adjoint(S)))
        #             U = tf.linalg.LinearOperatorBlockDiag([U, tf.linalg.LinearOperatorFullMatrix(V)])
        #         # convert unit
        #         self.unitary = tf.convert_to_tensor(U.to_dense())

        #     else:
        #         # use symmetric map to compute multi photon unitary
        #         S = tf.constant(symmetric_map(
        #             self.modes, self.photons), dtype=tf.complex64)
        #         # map to product state then use symmetric isometry to reduce to isomorphic subspace
        #         self.unitary = tf.matmul(S, tf.matmul(tf_multikron(self.unitary, self.photons), tf.linalg.adjoint(S)))

        # call build method of super class
        super(ULayer, self).build(input_shape)

    def call(self, inputs):
        """
        This is where the layer's logic lives.
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            A tensor or list/tuple of tensors.
        """
        # construct single photon unitary
        self.unitary = tf_clements_stitch(self.bms_spec, self.theta, self.phi, self.diag)

        # perform matrix calculation using fast einsum
        out = tf.einsum('ij,bj->bi', self.unitary, inputs, name="Einsum_left")

        return out

    def invert(self):
        """
        Computes the inverse of the unitary operator
        """
        # compute transpose of unitary
        self.unitary = tf.tranpose(self.unitary, conjugate=True, name="dagger_op")


    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(ULayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def output_size(self):
        return [self.output_dim, self.output_dim]


class PerformanceCallBack(tf.keras.callbacks.Callback):
    """
    Tensorflow callback to track performance metrics of network 
    """

    def __init__(self, loss=False, metric=False, name=None):
        super(PerformanceCallBack, self).__init__()
        if name is not None:
            self.name = name
        else:
            self.name = 'test'
        # best_weights to store the weights at which the minimum loss occurs.
        self.performance_dict = {}
        self.performance_dict['cosines'] = []
        self.performance_dict['loss'] = []

    def on_train_batch_end(self, batch, logs=None):
        # store log information
        self.performance_dict['cosines'].append(logs['cosine_similarity'])
        self.performance_dict['loss'].append(logs['loss'])

        # end training if metric is reached
        if np.mean(self.performance_dict['cosines'][-5:])>=0.84:
            self.model.stop_training = True
            # save info 
            np.save('cosine_{}.npy'.format(self.name), self.performance_dict['cosines'])
            np.save('loss_{}.npy'.format(self.name), self.performance_dict['loss'])


    def on_train_end(self, logs=None):
        # save info 
        np.save('Archive/cosine_{}.npy'.format(self.name), self.performance_dict['cosines'])
        np.save('Archive/loss_{}.npy'.format(self.name), self.performance_dict['loss'])



def train_many(num=100, advanced_mode=False, modes=6, photons=3, epochs=5):
    """
    Run many instances of the same network to generate some statistics
    """
    for i in tqdm(range(num)):
        # ------------------------------------------------------------
        # Section 0: Program parameter definitions
        # ------------------------------------------------------------
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
                          callbacks=[PerformanceCallBack(name='advanced_{}'.format(i))])




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
                          callbacks=[PerformanceCallBack(name='simple_{}'.format(i))])#model_checkpoint_callback,








def pretty_plot():
    """
    Plots data in a mildly pleasant fashion. 
    """

    # get some data
    advanced_data = np.load('performance_advanced.npy')
    simple_data = np.load('cosine_simple.npy')


    plt.plot(simple_data, 'b-.')
    plt.plot(advanced_data, 'r--')
    plt.axhline(y=0.84, color='k', linestyle='--', alpha=0.4)
    plt.grid(True)
    plt.xlabel('Training Steps')
    plt.ylabel('Performance (%)')
    plt.legend(['Uniform random','Subspace projection'])
    plt.title('Initialisation comparison for 111-dimensional input space')
    plt.show()








if __name__ == '__main__':
    train_many(advanced_mode=False, num=100, modes=6, photons=3, epochs=100)
    #pretty_plot()