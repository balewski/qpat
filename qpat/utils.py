import numpy as np
from itertools import tee

def convert_qubit_to_spherical ( a, b ):
    theta = 2 * np.arccos( a )
    phi   = 0
    if math.fmod( theta, np.pi ) != 0:
        t = b / np.sin( theta / 2 )
        if np.angle( t ) != -np.pi:
            phi = -1j * np.log( t )

    return [ theta, phi, 1 ]

def convert_spherical_to_cartesian ( theta, phi, r ):
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return [ x, y, z ]

def convert_qubit_to_cartesian ( a, b ):
    theta, phi, r = convert_qubit_to_spherical( a, b )
    return convert_spherical_to_cartesian( theta, phi, r )

def generate_haar_distribution ( num_qubits, trials = 100000 ):
	assert( num_qubits > 0 )

	dim  = 2 ** num_qubits
	haar = []

	for x in range( trials ):
		haar.append( unitary_group.rvs( dim ) )

	return haar

def pairwise ( iterable ):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee( iterable )
    next( b, None )
    return zip( a, b )