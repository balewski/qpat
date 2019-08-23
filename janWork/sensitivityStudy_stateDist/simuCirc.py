#!/usr/bin/env python

"""
Simulates one circuit with fixed number of sources and fixed nois amplitude (sigTheta)
saves results as yaml, includes circuit and other meta data
"""

import time,os
import itertools as it
import numpy     as np
import ruamel.yaml  as yaml

from qiskit import *
from qpat   import *
from pprint import pprint

from Plotter_CircSensitivity import Plotter_CircSensitivity

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity", default=1,choices=[0, 1, 2,3],
                        help="increase output verbosity", dest='verb')
    parser.add_argument( "-X","--noXterm", dest='noXterm',
                         action='store_true', default=False,
                         help="disable X-term for batch mode")
    parser.add_argument("-d","--dataPath",help="path to input",
                        default='data')
    parser.add_argument("-o","--outPath",
                        default='out',help="output path for plots and tables")
    parser.add_argument("-c","--circName",
                        default='hidStr_2q',help="name of circuit")
    parser.add_argument("-n", "--events", type=int, default=1000,
                        help="events for single experiment")
    parser.add_argument("-s","--sigTheta",type=float,
                        default=0.2,help="width of noise distribution, in radians")
    parser.add_argument("-m","--noiseMult",type=int,
                        default=1,help="mutiplicity of noise sources per experiment")

    args = parser.parse_args()
    args.prjName='qpadEd'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args



#= = = = = = = = = = = = = = = = 
def noise_injection ( program, simulator, sources, trials, sigTheta ):
    """
    Injects noise into a program and measures the result when simulated.
    Test all possible configurations of $sources noise sources.

    Args:
        program (QuantumCircuit):      The program to inject noise into.

        simulator ((QuantumCircuit | np.matrix) -> Result)The quantum simulator used.

        measure  The metric function that compares two results.

        sources (Natural Number)   The number of noise sources to test.

        trials (Natural Number)   The number of trials for each configuration of noise sources.

        sigTheta (rad)   width of angle noise

        samplingType (string)  How noise is sampled, see qpat for more information.

    Returns:
        A map from tuples of gates to distribution of results.
    """
    measure=state_fidelity
    samplingType = "gauss2"  # added by Jan
    assert( isinstance( program, QuantumCircuit ) )
    assert( 0 < sources )
    assert( 0 < trials )

    output = {}
    perfect_results = simulator( program )
    gate_pos_array  = get_gate_pos( program )

    # combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
    experimentL=[ x for x  in it.combinations_with_replacement( gate_pos_array, sources )]
    for gate_config in experimentL:
        #print('gc',gate_config)
        #if gate_config[0]==gate_config[1]: continue
        #if len(gate_config[0][1])==1 : continue
        programs_noisy = inject_noise( program, trials,[( gate_config, sigTheta )], samplingType )
        results_noisy  = simulator( programs_noisy )

        result_dist = measure( results_noisy, perfect_results )
        gates       = tuple( [ g[0] for g in gate_config ] )

        output[ gates ] = result_dist

    return output,experimentL

#...!...!..................
def read_yaml(ymlFn):
        print('  read  yaml:',ymlFn,' ',end='')
        start = time.time()
        ymlFd = open(ymlFn, 'r')
        bulk=yaml.load( ymlFd, Loader=yaml.CLoader)
        ymlFd.close()
        print(' done, size=%d'%len(bulk),'  elaT=%.1f sec'%(time.time() - start))
        return bulk

#...!...!..................
def write_yaml(rec,ymlFn,verb=1):
        start = time.time()
        ymlFd = open(ymlFn, 'w')
        yaml.dump(rec, ymlFd, Dumper=yaml.CDumper)
        ymlFd.close()
        xx=os.path.getsize(ymlFn)/1024
        if verb:
                print('  closed  yaml:',ymlFn,' size=%.2f kB'%xx,'  elaT=%.1f sec'%(time.time() - start))




#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    plot=Plotter_CircSensitivity(args)
    
    circF=args.dataPath+'/'+args.circName+'.qasm'
    circOrg=QuantumCircuit.from_qasm_file( circF )
    print('\ncirc original: ', circF); print(circOrg,'\n')
    circCompl=compile_circuit( circOrg )
    print('\ncirc compiled:'); print(circCompl,'\n')
    
    gatePosInfo=get_gate_pos_info(circCompl)
    print('gate position info')
    pprint(gatePosInfo)

    gate_posL  = get_gate_pos( circCompl )
    print('gate_posL=',gate_posL)
    numEve=args.events
    
    startT=time.time()
    dataD,experimentL = noise_injection( circCompl, statevector_simulator,
                            args.noiseMult,numEve, args.sigTheta)
    posL=sorted(dataD.keys())
    elaT=time.time()-startT
    print('done %d eve, %d positions, etaT=%.1f mon'%(numEve, len(posL),elaT/60.))
    print('combinations_with_replacement gateCnt=%d x noiseMult=%d --> %d noise experiments'%(len(gate_posL),args.noiseMult,len(experimentL)))
    
    print('dataD=',posL)
    distV=[]; meanV=[]; stderV=[]
    maxDist=0
    # convert fidelity to distance
    for pos in posL:
        fidel1=np.array(dataD[pos])
        # remove  bad fidelity>1
        fidel1=fidel1[fidel1<1.0]
        data1=np.sqrt(1-fidel1)
        #print('pos=',pos,'len:',len(fidel1),type(fidel1))
        meanJan=np.mean(data1)
        errJan=np.std(data1)/np.sqrt(numEve)
        xx=np.max(data1)
        if maxDist < xx: maxDist=xx
        meanV.append(float(meanJan)); stderV.append(float(errJan)); distV.append(data1)
        print('pos=',pos,', meanJan=%.3f +/- %.4f eve=%d'%(meanJan,errJan,data1.shape[0]))

    print('sigTheta=%.3f, numEve=%d'%(args.sigTheta,numEve))

    qasmD={'oryginal':circOrg.qasm().split('\n'), 'compiled':circCompl.qasm().split('\n')}
    metaD={'sigTheta':args.sigTheta,'gatePosInfo':gatePosInfo,'numEve':numEve,'elaTime':elaT,'circName':args.circName,'qasm':qasmD}
    resultD={'mean':meanV,'stder':stderV,'pos':posL}
    outD={'meta':metaD,'results':resultD}
    write_yaml(outD,args.outPath+'/'+args.circName+'.stateDist.yaml')
    resultD['fidel']=distV
    exit(99)
    plot.stateDistance_average(resultD, metaD,tit='ex2')
    plot.stateDistance_spectra(resultD, metaD)
    plot.display_all(args,args.circName)
    #plt=plot_
    #plot_
    #plt.tight_layout(); plt.show()
