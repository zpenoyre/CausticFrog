#runs the cython simulation (jury is out on whether this need be it's own file...)
import Cython
import pyximport
pyximport.install()
from . import cythonSim

#this allows line profiler (via ipython magic %lprun) to profile cython functions
from Cython.Compiler.Options import get_directive_defaults

get_directive_defaults()['linetrace'] = True
get_directive_defaults()['binding'] = True

def runSim(nShells,nPhase,nEcc,T,dt,rMin,rMax,name,nOutput,dmMass,baryonInit,baryonMass,findEcc,G=4.96e-15):
    cythonSim.updateGlobal(G) #updates global variables
    cythonSim.setFunctions(dmMass,baryonInit,baryonMass,findEcc) #sets user definied function for the intial state and final baryon mass
    cythonSim.runSim(nShells,nPhase,nEcc,T,dt,rMin,rMax,nOutput,name) #runs the simulation