# ##############################################################################
# Measures come in three types
# 1. A comparison between quantum states
# 2. A comparison between quantum processes
# 3. A comparison between distributions of classical states
# ##############################################################################

import numpy as np

# 1. A comparison between quantum states
def state_fidelity ( state1, state2 ):
	assert( len( state1 ) == len( state2 ) )
	return np.square( np.abs( np.dot( state1.conj().transpose(), state2 ) ) )

# 2. A comparison between quantum processes
def process_fidelity ( process1, process2 ):
	p1 = np.asarray( process1, dtype = 'complex128' )
	p2 = np.asarray( process2, dtype = 'complex128' )

	assert( p1.shape == p2.shape )

	x1 = np.abs( np.trace( np.dot( p1.conj().transpose(), p2 ) ) ) ** 2
	x2 = len( p1 ) ** 2
	return x1 / x2

# 3. A comparison between distributions of classical states
def total_variation_distance ( dist1, dist2 ):
	assert( len( dist1 ) == len( dist2 ) )
	return (.5) * np.sum( np.abs( dist1 - dist2 ) )

# 3. A comparison between distributions of classical states
def kl_divergence ( dist1, dist2 ):
	dist1 = np.asarray( dist1, dtype = 'float128' )
	dist2 = np.asarray( dist2, dtype = 'float128' )

	assert( len( dist1 ) == len( dist2 ) )

	sum = 0.
	for i in range( len( dist1 ) ):
		if dist1[i] == 0 or dist2[i] == 0:
			continue

		sum += dist1[i] * ( np.log( dist1[i] ) - np.log( dist2[i] ) )
	return sum
