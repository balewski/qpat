# ##############################################################################
# A quantum simulator will simulate the quantum program passed as an argument.
# Three Types of Simulators
# 1. A simulator that returns a unitary matrix
# 2. A simulator that returns the state vector output
#     when run with specified input.
# 3. A simulator that returns the probability distribution of classical
#     outputs when run with specified input.
# ##############################################################################

import numpy as np
from qiskit import BasicAer, execute, QuantumCircuit

# 1. A simulator that returns a unitary matrix
def unitary_simulator ( program ):
	if isinstance( program, QuantumCircuit ):
		unitary_backend = BasicAer.get_backend( 'unitary_simulator' )
		unitary_job     = execute( program, unitary_backend )
		unitary_results = unitary_job.result()
		unitary_matrix  = unitary_results.get_unitary( program )
		return unitary_matrix

	elif isinstance( program, np.ndarray ):
		return program

# 2. A simulator that returns the state vector output
#     when run with specified input.
def statevector_simulator ( program, init_vector = [] ):
	if isinstance( program, QuantumCircuit ):
		statevector_backend = BasicAer.get_backend( 'statevector_simulator' )
		statevector_job = None

		if init_vector != []:
			statevector_job = execute( program, statevector_backend,
							  backend_options = {
							  		'initial_statevector': np.array( init_vector )
							  		} )
		else:
			statevector_job = execute( program, statevector_backend )

		statevector_results = statevector_job.result()
		statevector_vector  = statevector_results.get_statevector( program )
		return statevector_vector

	elif isinstance( program, np.ndarray ):
		if init_vector == []:
			init_vector = np.zeros( program.shape[-1], dtype = 'clongdouble' )
			init_vector[0] = 1
		return np.matmul( program, init_vector.T )

# 3. A simulator that returns the probability distribution of classical
#     outputs when run with specified input.
def prob_dist_simulator ( program, init_vector = [] ):
	state_vector = statevector_simulator( program, init_vector )
	state_vector = np.asarray( state_vector, dtype = 'clongdouble' )

	assert( state_vector.ndim <= 2 )

	if state_vector.ndim == 1:
		return np.square( np.abs( state_vector ) )

	else:
		stack_dist = []
		for vector in state_vector:
			stack_dist.append( np.square( np.abs( vector ) ) )
		return np.asarray( stack_dist, dtype = 'longdouble' )