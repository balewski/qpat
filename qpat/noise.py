import numpy as np
import scipy.stats
import copy

from .utils import pairwise
from .simulator import unitary_simulator
from itertools import product

# ##############################################################################
# Noise is represented as a random unitary rotation
# TODO: Better Description
# Main func:
# ##############################################################################

sx = np.matrix( [[0, 1  ], [1 , 0 ]], dtype = 'clongdouble' )
sy = np.matrix( [[0, -1j], [1j, 0 ]], dtype = 'clongdouble' )
sz = np.matrix( [[1, 0  ], [0 , -1]], dtype = 'clongdouble' )

paulis  = [ [ sx, sy, sz ] ]

def paulis_get ( i, qubits = 1 ):
    assert( qubits > 0 )
    assert( i < 3 ** qubits )

    # If we haven't generated the Pauli's for this dimension
    # we generate them
    while len( paulis ) < qubits:
        paulis.append( [ np.kron( si, sj ) for si, sj in product( paulis[0], paulis[-1] ) ] )

    return paulis[qubits-1][i]

def paulis_dot ( n, qubits = 1 ):
    assert( qubits > 0 )
    assert( len( n ) == 3 ** qubits )

    paulisum = np.zeros( (2 ** qubits, 2 ** qubits), dtype = 'clongdouble' )
    for i in range( len( n ) ):
        paulisum += n[i] * paulis_get( i, qubits )
    return paulisum

def construct_noise_operator ( n, theta, qubits = 1 ):
    assert( theta > 0 and theta <= np.pi )
    assert( qubits > 0 )
    assert( len( n ) == 3 ** qubits )

    return np.identity(2 ** qubits) * np.cos(theta/2) - (0+1j) * paulis_dot(n, qubits) * np.sin(theta/2)

def sample_unit_vectors ( num, dim = 3 ):
    assert( num > 0 )
    assert( dim > 0 )

    vec  = np.random.randn( dim, num )
    vec /= np.linalg.norm ( vec, axis = 0 )
    return vec.T

def sample_noise_operators_gauss ( strength_factor, qubits = 1, num = 1000 ):
    assert( strength_factor > 0 and strength_factor <= 1 )
    assert( qubits > 0 )
    assert( num > 0 )

    noise_ops    = []
    theta_std    = strength_factor * np.pi
    vector_dim   = 3 ** qubits
    unit_vectors = sample_unit_vectors( num, vector_dim )

    for n in unit_vectors:
        theta = scipy.stats.truncnorm.rvs( 0, 1 ) * theta_std
        noise_ops.append( construct_noise_operator( n, theta, qubits ) )

    return np.stack( noise_ops )

def sample_noise_operators_max ( strength_factor, qubits = 1, num = 1000 ):
    assert( strength_factor > 0 and strength_factor <= 1 )
    assert( qubits > 0 )
    assert( num > 0 )

    noise_ops    = []
    theta_max    = strength_factor * np.pi
    vector_dim   = 3 ** qubits
    unit_vectors = sample_unit_vectors( num, vector_dim )

    for n in unit_vectors:
        theta = np.random.uniform( high = theta_max )
        noise_ops.append( construct_noise_operator( n, theta, qubits ) )

    return np.stack( noise_ops )

def sample_noise_operators_exact ( strength_factor, qubits = 1, num = 1000 ):
    assert( strength_factor > 0 and strength_factor <= 1 )
    assert( qubits > 0 )
    assert( num > 0 )

    noise_ops    = []
    theta_exact  = strength_factor * np.pi
    vector_dim   = 3 ** qubits
    unit_vectors = sample_unit_vectors( num, vector_dim )

    for n in unit_vectors:
        theta = theta_exact
        noise_ops.append( construct_noise_operator( n, theta, qubits ) )

    return np.stack( noise_ops )

def sample_noise_operators_min ( strength_factor, qubits = 1, num = 1000 ):
    assert( strength_factor > 0 and strength_factor <= 1 )
    assert( qubits > 0 )
    assert( num > 0 )

    noise_ops    = []
    theta_min    = strength_factor * np.pi
    vector_dim   = 3 ** qubits
    unit_vectors = sample_unit_vectors( num, vector_dim )

    for n in unit_vectors:
        theta = np.random.uniform( low = theta_min, high = np.pi )
        noise_ops.append( construct_noise_operator( n, theta, qubits ) )

    return np.stack( noise_ops )

swap = np.matrix( [ [1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1] ] )

def extend_noise_operators ( noise_ops, target_qubits, num_qubits ):
    assert( len( noise_ops ) > 0 )
    assert( len( target_qubits ) > 0 )
    assert( all( [ qubit >= 0 for qubit in target_qubits ] ) )
    assert( all( [ qubit < num_qubits for qubit in target_qubits ] ) )

    # Extend Noise Operators
    dim = num_qubits - len( target_qubits )
    noise_ops = np.kron( noise_ops, np.identity( 2 ** dim ) )

    # Places Necessary Swaps
    target_qubits.sort( reverse = True )

    left  = len( target_qubits ) - 1

    for initial, target in enumerate( target_qubits ):
        initiali = len( target_qubits ) - 1 - initial

        distance = target - initiali
        right    = num_qubits - target - 1

        if distance > 0:
            x = swap
            y = swap
            for i in range( 1, distance ):
                x = np.kron( np.identity(2), x )
                y = np.kron( y, np.identity(2) )
                x = np.matmul( y, x )
            x = np.kron( x, np.identity(2 ** right) )
            pre_permutation_op = np.asarray( np.kron( np.identity(2 ** left), x ) )

            x = swap
            y = swap
            for i in range( 1, distance ):
                x = np.kron( x, np.identity(2) )
                y = np.kron( np.identity(2), y )
                x = np.matmul( y, x )
            x = np.kron( x, np.identity(2 ** right) )
            post_permutation_op = np.asarray( np.kron( np.identity(2 ** left), x ) )

            noise_ops = np.matmul( noise_ops, pre_permutation_op )
            noise_ops = np.matmul( post_permutation_op, noise_ops )

        left -= 1

    return noise_ops

def generate_noise_operators ( strength_factor, target_qubits, num_qubits, num = 1000, sampling = 'max' ):
    noise_ops = None
    if sampling == 'max':
        noise_ops = sample_noise_operators_max( strength_factor, len( target_qubits ), num )
    elif sampling == 'min':
        noise_ops = sample_noise_operators_min( strength_factor, len( target_qubits ), num )
    elif sampling == 'exact':
        noise_ops = sample_noise_operators_exact( strength_factor, len( target_qubits ), num )
    elif sampling == 'gauss':
        noise_ops = sample_noise_operators_gauss( strength_factor, len( target_qubits ), num )
    noise_ops = extend_noise_operators( noise_ops, target_qubits, num_qubits )
    return noise_ops

def inject_noise ( program, num, noise_defs, sampling = 'max' ):
    all_noise_pos = []

    for noise_def in noise_defs:
        all_noise_pos += [ x for x in noise_def[0] if x not in all_noise_pos ]

    # Slice the program
    program_slices = {}

    program_slice_pos = set( [ n[0] for n in all_noise_pos ] )
    program_slice_pos.add( -1 )
    program_slice_pos.add( len(program.data) )

    for slice_from, slice_to in pairwise( sorted( program_slice_pos ) ):
        slice_from = 0 if slice_from == -1 else slice_from
        program_copy      = copy.deepcopy( program )
        program_copy.data = program_copy.data[ slice_from : slice_to ]
        if len( program_copy.data ) != 0:
            program_slices[ slice_from ] = ( unitary_simulator( program_copy ) )

    # Insert error and accumulate program
    results = np.stack( [ np.identity( 2 ** program.width() ) for i in range( num ) ] )

    if 0 not in program_slice_pos:
        results = np.matmul( program_slices[0], results )

    for index, qubits in sorted( all_noise_pos ):
        errors = None

        for noise_def in noise_defs:
            if (index, qubits) in noise_def[0]:
                errors = generate_noise_operators( noise_def[1], qubits, program.width(), num, sampling )
                results = np.matmul( errors, results )

        if index in program_slices:
            results = np.matmul( program_slices[index], results )

    return results