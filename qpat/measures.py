# ##############################################################################
# Measures come in three types
# 1. A comparison between quantum states
# 2. A comparison between quantum processes
# 3. A comparison between distributions of classical states
# ##############################################################################

import numpy as np

# 1. A comparison between quantum states
def state_fidelity ( state1, state2 ):
	s1 = np.asarray( state1, dtype = 'clongdouble' )
	s2 = np.asarray( state2, dtype = 'clongdouble' )

	assert( s1.shape[-1] == s2.shape[-1] )
	assert( np.abs( s1.ndim - s2.ndim ) <= 1 )

	if s1.ndim == s2.ndim:
		return np.square( np.abs( np.dot( s1.conj().transpose(), s2 ) ) )

	elif s1.ndim > s2.ndim:
		data = []
		for s1v in s1:
			data.append( np.square( np.abs( np.dot( s1v.conj().transpose(), s2 ) ) ) )
		return data

	elif s1.ndim < s2.ndim:
		data = []
		for s2v in s2:
			data.append( np.square( np.abs( np.dot( s1.conj().transpose(), s2v ) ) ) )
		return data

# 2. A comparison between quantum processes
def process_fidelity ( process1, process2 ):
	p1 = np.asarray( process1, dtype = 'clongdouble' )
	p2 = np.asarray( process2, dtype = 'clongdouble' )

	assert( p1.shape[-2:-1] == p2.shape[-2:-1] )
	assert( np.abs( p1.ndim - p2.ndim ) <= 1 )

	if p1.ndim == p2.ndim:
		x1 = np.abs( np.trace( np.dot( p1.conj().transpose(), p2 ) ) ) ** 2
		x2 = len( p1 ) ** 2
		return x1 / x2

	elif p1.ndim > p2.ndim:
		data = []
		for p1v in p1:
			x1 = np.abs( np.trace( np.dot( p1v.conj().transpose(), p2 ) ) ) ** 2
			x2 = len( p1v ) ** 2
			data.append( x1 / x2 )
		return np.asarray( data, dtype = 'longdouble' )

	elif p1.ndim < p2.ndim:
		data = []
		for p2v in p2:
			x1 = np.abs( np.trace( np.dot( p1.conj().transpose(), p2v ) ) ) ** 2
			x2 = len( p1 ) ** 2
			data.append( x1 / x2 )
		return np.asarray( data, dtype = 'longdouble' )


# 3. A comparison between distributions of classical states
def total_variation_distance ( dist1, dist2 ):
	d1 = np.asarray( dist1, dtype = 'longdouble' )
	d2 = np.asarray( dist2, dtype = 'longdouble' )

	assert( d1.shape[-1] == d2.shape[-1] )
	assert( np.abs( d1.ndim - d2.ndim ) <= 1 )

	if d1.ndim == d2.ndim:
		return (.5) * np.sum( np.abs( d1 - d2 ) )

	elif d1.ndim > d2.ndim:
		data = []
		for d1v in d1:
			data.append( (.5) * np.sum( np.abs( d1v - d2 ) ) )
		return np.asarray( data, dtype = 'longdouble' )

	elif d1.ndim < d2.ndim:
		data = []
		for d2v in d2:
			data.append( (.5) * np.sum( np.abs( d1 - d2v ) ) )
		return np.asarray( data, dtype = 'longdouble' )

# 3. A comparison between distributions of classical states
def element_wise_variation_distance ( dist1, dist2 ):
	d1 = np.asarray( dist1, dtype = 'longdouble' )
	d2 = np.asarray( dist2, dtype = 'longdouble' )

	assert( d1.shape[-1] == d2.shape[-1] )
	assert( np.abs( d1.ndim - d2.ndim ) <= 1 )

	if d1.ndim == d2.ndim:
		return np.abs( d1 - d2 )

	elif d1.ndim > d2.ndim:
		data = []
		for d1v in d1:
			data.append( np.abs( d1v - d2 ) )
		return np.asarray( data, dtype = 'longdouble' )

	elif d1.ndim < d2.ndim:
		data = []
		for d2v in d2:
			data.append( np.abs( d1 - d2v ) )
		return np.asarray( data, dtype = 'longdouble' )

# 3. A comparison between distributions of classical states
def kl_divergence ( dist1, dist2 ):
	dist1 = np.asarray( dist1, dtype = 'longdouble' )
	dist2 = np.asarray( dist2, dtype = 'longdouble' )

	assert( len( dist1 ) == len( dist2 ) )

	sum = 0.
	for i in range( len( dist1 ) ):
		if dist1[i] == 0 or dist2[i] == 0:
			continue

		sum += dist1[i] * ( np.log( dist1[i] ) - np.log( dist2[i] ) )
	return sum
