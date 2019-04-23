from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer

from qiskit.transpiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation

from math import pi

def compile_circuit ( circuit ):
	pm = PassManager()
	pm.append( CXCancellation() )
	pm.append( Optimize1qGates() )
	pm.append( CXCancellation() )
	pm.append( Optimize1qGates() )
	pm.append( CXCancellation() )
	pm.append( Optimize1qGates() )
	return transpile( circuit, BasicAer.get_backend( 'unitary_simulator' ), pass_manager = pm )

def pretty_print_circuit ( circuit ):
	circuit.draw( output='mpl', plot_barriers=False, style={'fold': 26, 'usepiformat': True, 'showindex': True, 'compress': False} )

def generate_toffoli():
	qreg = QuantumRegister( 3, 'q' )
	circ = QuantumCircuit ( qreg, name = "Toffoli" )

	circ.h( qreg[2] )
	circ.cx( qreg[1], qreg[2] )
	circ.tdg( qreg[2] )
	circ.cx( qreg[0], qreg[2] )
	circ.t( qreg[2] )
	circ.cx( qreg[1], qreg[2] )
	circ.tdg( qreg[2] )
	circ.cx( qreg[0], qreg[2] )
	circ.barrier()
	circ.t( qreg[1] )
	circ.t( qreg[2] )
	circ.cx( qreg[0], qreg[1] )
	circ.h( qreg[2] )
	circ.barrier()
	circ.tdg( qreg[1] )
	circ.t( qreg[0] )
	circ.cx( qreg[0], qreg[1] )

	return circ

def generate_qft ( num_qubits = 3 ):
	assert( num_qubits > 0 )

	qreg = QuantumRegister( num_qubits, 'q' )
	circ = QuantumCircuit ( qreg, name = "QFT" )

	for i in range( num_qubits ):
		for j in range( i ):
			circ.cu1( pi / float( 2**(i-j) ), qreg[i], qreg[j] )
		circ.h( qreg[i] )

	return circ

def generate_ghz ( num_qubits = 3 ):
	assert( num_qubits > 0 )

	qreg = QuantumRegister( num_qubits, 'q' )
	circ = QuantumCircuit ( qreg, name = "GHZ" )

	circ.h( qreg[0] )

	for i in range( 1, num_qubits ):
		circ.cx( qreg[i - 1], qreg[i] )

	return circ

def generate_bv():
	qreg = QuantumRegister( 4, 'q' )
	circ = QuantumCircuit ( qreg, name = "BV" )

	circ.x( qreg[3] )
	for i in range(4):
		circ.h( qreg[i] )
	for i in range(3):
		circ.cx( qreg[i], qreg[3] )
	for i in range(4):
		circ.h( qreg[i] )

	return circ

def generate_fredkin():
	qreg = QuantumRegister( 3, 'q' )
	circ = QuantumCircuit ( qreg, name = "Fredkin" )

	circ.x( qreg[0] )
	circ.x( qreg[1] )
	circ.cx( qreg[2], qreg[1] )
	circ.cx( qreg[0], qreg[1] )
	circ.h( qreg[2] )
	circ.t( qreg[0] )
	circ.tdg( qreg[1] )
	circ.t( qreg[2] )
	circ.cx( qreg[2], qreg[1] )
	circ.cx( qreg[0], qreg[2] )
	circ.t( qreg[1] )
	circ.cx( qreg[0], qreg[1] )
	circ.tdg( qreg[2] )
	circ.tdg( qreg[1] )
	circ.cx( qreg[0], qreg[2] )
	circ.cx( qreg[2], qreg[1] )
	circ.t( qreg[1] )
	circ.h( qreg[2] )
	circ.cx( qreg[2], qreg[1] )

	return circ

# A 3 qubit grover's algorithm that searches for (1,1)
def generate_grovers():
	qreg = QuantumRegister( 3, 'q' )
	circ = QuantumCircuit ( qreg, name = "Grover" )

	circ.x( qreg[0] )
	circ.h( qreg[1] )
	circ.h( qreg[2] )
	circ.h( qreg[0] )
	circ.h( qreg[0] )
	circ.cx( qreg[1], qreg[0] )
	circ.tdg( qreg[0] )
	circ.cx( qreg[2], qreg[0] )
	circ.t( qreg[0] )
	circ.cx( qreg[1], qreg[0] )
	circ.tdg( qreg[0] )
	circ.cx( qreg[2], qreg[0] )
	circ.t( qreg[0] )
	circ.tdg( qreg[1] )
	circ.h( qreg[0] )
	circ.cx( qreg[2], qreg[1] )
	circ.tdg( qreg[1] )
	circ.cx( qreg[2], qreg[1] )
	circ.s( qreg[1] )
	circ.t( qreg[2] )
	circ.h( qreg[1] )
	circ.h( qreg[2] )
	circ.x( qreg[1] )
	circ.x( qreg[2] )
	circ.h( qreg[1] )
	circ.cx( qreg[2], qreg[1] )
	circ.h( qreg[1] )
	circ.x( qreg[1] )
	circ.x( qreg[2] )
	circ.h( qreg[1] )
	circ.h( qreg[2] )

	return circ
