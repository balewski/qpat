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
from qiskit import BasicAer, execute

# 1. A simulator that returns a unitary matrix
def unitary_simulator ( program ):
	unitary_backend = BasicAer.get_backend( 'unitary_simulator' )
	unitary_job     = execute( program, unitary_backend )
	unitary_results = unitary_job.result()
	unitary_matrix  = unitary_results.get_unitary( program )
	return unitary_matrix

# 2. A simulator that returns the state vector output
#     when run with specified input.
def statevector_simulator ( program, init_vector = [] ):
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

# 3. A simulator that returns the probability distribution of classical
#     outputs when run with specified input.
def prob_dist_simulator ( program, init_vector = [] ):
	state_vector  = statevector_simulator( program, init_vector )
	psuedo_counts = {}
	for i, element in enumerate( state_vector ):
		format_str = "{0:0" + str(program.width()) + "b}"
		bit_string = str( format_str.format( i ) )
		value = np.square( np.abs( element ) )
		if value != 0:
			psuedo_counts[bit_string] = value

	return psuedo_counts