# The cython simulation itself
import numpy as np
import scipy.optimize
#cimport numpy as np (pyximport seems to break on this! not needed anyway)

cdef edgeClass[:] edges
cdef shellClass[:] shells

cdef double G=1
cpdef void updateGlobal(new_G):
    global G
    G=new_G

# Basic function which mass enclosed functions used elsewhere are mapped onto (see http://docs.cython.org/en/latest/src/tutorial/cdef_classes.html)
cdef class massFunction:
    cpdef double evaluate(self,double r) except *:
        return 0

# these functions are called in density checks
cdef massFunction dmMass
cdef massFunction baryonInit
cdef massFunction baryonMass

# this function is the distribution of ccentricities (later probably to be replaced with vr and vt)
cdef massFunction findEcc
def findPsi(P,ecc): #function needed to find psi for a given eccentricity and probability
    def P_eta(eta,P,ecc):
        return ((eta-ecc*np.sin(eta))/(2*np.pi))-P
    eta=scipy.optimize.brentq(P_eta,0,2*np.pi,args=(P,ecc))
    return np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(eta))

# and their form is inputted from python here
cpdef void setFunctions(massFunction dm, massFunction iBaryon, massFunction fBaryon, massFunction eDist):
    global dmMass, baryonInit, baryonMass, findEcc
    dmMass=dm
    baryonInit=iBaryon
    baryonMass=fBaryon
    findEcc=eDist
    
#object based shells
cdef class edgeClass: #edges of shells (radii being evolved)
    cdef public:
        double r, e, psi, lSq, vr, M
        int ind, lowShell, highShell
    def __cinit__(self):
        self.r=0
        self.lSq=0
        self.vr=0 #for vt just take l/r
        self.lowShell=-1 #index of shell between the last edge and this
        self.highShell=-1 #index of shell between this edge and the next
        self.M=0 #mass within this shell (by radius)
        #self.inside=np.array([-1]) #list of edge pairs contained entirely within this edge, gives index of first edge
        self.ind=-1 #for edges[whichEdge] should equal whichEdge
    cpdef void updateR(self,dt):
        self.r+=self.vr*dt
    cpdef void updateV(self,dt):
        self.vr+=((self.r**-3)*(self.lSq-G*self.M*self.r))*dt
        
        
cdef class shellClass: #shells themselves (constant mass, changing density)
    cdef public:
        double m, rho
        int lowEdge, highEdge, ind
    def __cinit__(self):
        self.m=0
        self.rho=0
        self.lowEdge=-1 #index of first edge
        self.highEdge=-1 #index of second edge
        self.ind=-1 #for shells[whichShell] should equal whichShell

cdef void findMass():
    cdef:
        int i,count,nOverlap
        double dV, fourPi_three=4*np.pi/3
        int nShell=len(shells)
        int nEdge=len(edges)
    rad=np.zeros(nEdge,dtype=float)
    shellFlag=np.ones(nShell,dtype=int) #0 if shell already enclosed/overlapping
    for i in range(nEdge):
        rad[i]=edges[i].r
        
    sort=np.argsort(rad)
    #print('sorted order: ',sort)
    cdef list overlap=[] #list of shells currently overlapping
    
    cdef edgeClass thisEdge=edges[sort[0]]
    cdef double mass=baryonMass.evaluate(thisEdge.r)
    thisEdge.M=mass
    cdef shellClass thisShell
    
    #print('________________First edge:')
    #printEdge(sort[0])
    
    if(thisEdge.lowShell !=-1):
        thisShell=shells[thisEdge.lowShell]
        shellFlag[thisShell.ind]=0
        overlap.append([thisShell.lowEdge,thisShell.rho])
    
    if(thisEdge.highShell !=-1):
        thisShell=shells[thisEdge.highShell]
        shellFlag[thisShell.ind]=0
        overlap.append([thisShell.highEdge,thisShell.rho])
    
    cdef edgeClass lastEdge=thisEdge
    for i in range(1,nEdge):
        #print(i,'th edge: _________________')
        #printEdge(sort[i])
        
        thisEdge=edges[sort[i]]
        dV=fourPi_three*(thisEdge.r**3 - lastEdge.r**3)
        mass+=baryonMass.evaluate(thisEdge.r)-baryonMass.evaluate(lastEdge.r)
        #print('shell mass: ',baryonMass.evaluate(lastEdge.r,thisEdge.r))
        nOverlap=len(overlap)
        count=0
        while (count<nOverlap):
            #print('for entry: ',i,', shell: ',sort[i],' has overlap: ',overlap[count][:],' of: ',nOverlap)
            mass+=dV*overlap[count][1]
            if (thisEdge.ind==overlap[count][0]):
                overlap.pop(count)
                nOverlap-=1
            else:
                count+=1
        #print('edge had mass: ',thisEdge.M,' which has changed by a factor: ',mass/thisEdge.M)
        thisEdge.M=mass
        
        whichShell=thisEdge.lowShell
        if ((shellFlag[whichShell]==1) & (thisEdge.lowShell !=-1)): #shell initially behind edge now overlapping with further out shells
            #print("Adding other side of this shell")
            #printShell(whichShell)
            shellFlag[shells[whichShell].ind]=0
            overlap.append([shells[whichShell].lowEdge,shells[whichShell].rho])
            
        whichShell=thisEdge.highShell
        if ((shellFlag[whichShell]==1) & (thisEdge.highShell !=-1)): #shell initially ahead of edge now overlapping with further out shells
            #print("Adding other side of this shell")
            #printShell(whichShell)
            shellFlag[shells[whichShell].ind]=0
            overlap.append([shells[whichShell].highEdge,shells[whichShell].rho])
            
        lastEdge=thisEdge
# END ___ findMass() ___________________
    
#initialises edges and, where they exist, the shells they contain (most properties blank)
def init(nShells,nEcc,nPhase,minR,maxR):
    nEdge=2*(nShells+1)*nPhase*nEcc
    nShell=2*nShells*nPhase*nEcc #total number of shells of all phases and e
    global edges
    global shells
    edgeArray=np.array([None]*nEdge)
    edges=edgeArray
    shellArray=np.array([None]*nShell)
    shells=shellArray
    
    # Initial radii of edges
    rs=np.linspace(np.sqrt(minR),np.sqrt(maxR),nShells+1)**2
    shellsCreated=0
    
    # Creates all the edges
    for i in range(nEdge):
        newEdge=edgeClass(i)
        
        newEdge.ind=i
        newEdge.r=rs[i%(nShells+1)]
        newEdge.M=baryonInit.evaluate(newEdge.r)+dmMass.evaluate(newEdge.r)
        #print('edge ',i,' has mass ',newEdge.M)
        #m0+(4*np.pi*rho*(newEdge.r**3 - rs[0]**3)/3)
        
        #print('P ecc: ',(int(i/(2*(nShells+1)*nPhase))/nEcc))
        e=findEcc.evaluate((int(i/(2*(nShells+1)*nPhase))/nEcc))
        #print('ecc: ',e)
        #1+small-np.sqrt(1-(int(i/(2*(nShells+1)*nPhase))/nEcc)) #linearly decreasing eccentricity probability
        #print('P phase: ',(int(i/(2*(nShells+1)))%nPhase)/nPhase)
        psi=findPsi((int(i/(2*(nShells+1)))%nPhase)/nPhase,e)+(int(i/(nShells+1))%2)*np.pi
        #print('psi: ',psi)
        a=newEdge.r*(1+e*np.cos(psi))/(1-e**2)
        newEdge.lSq=G*newEdge.M*a*(1-e**2)
        newEdge.vr=np.sqrt(newEdge.lSq)*e*np.sin(psi)/(a*(1-e**2))
        
        if (i%(nShells+1)==0): #nearest edges have no inner shell
            newEdge.lowShell=-1
        else:
            newEdge.lowShell=shellsCreated-1
        
        if (i+1)%(nShells+1)==0: #farthest edge has no next shell
            newEdge.highShell=-1
            edges[i]=newEdge
            continue
        
        newEdge.highShell=shellsCreated
        edges[i]=newEdge
        
        # Next shell exists so let's make it
        newShell=shellClass()
        
        newShell.ind=shellsCreated
        newShell.lowEdge=i
        newShell.highEdge=i+1
        lowR=rs[i%(nShells+1)]
        highR=rs[(i+1)%(nShells+1)]
        mShell=dmMass.evaluate(highR)-dmMass.evaluate(lowR)
        newShell.m=mShell/(2*nPhase*nEcc) #adjusted for overlapping shells
        newShell.rho=newShell.m/((4./3.)*np.pi*(highR**3 - lowR**3)) #average density over shell
        
        shells[shellsCreated]=newShell 
        shellsCreated+=1 
# END ___ init() ___________________
        
#outputs data for shells and their edges, sufficient for plotting and/or restart
def outputData(step,dt,whichStep,nOutput,simName):
    fName='output/'+simName+'.'+str(int(whichStep))+'_'+str(int(nOutput))+'.txt'
    headerString='step='+str(step)+', t='+str(step*dt)
    print('Saving to file: ',fName)
    # 0_shellIndex, 1_shellMass, 2_shellDens, 3_r1, 4_r2, 5_vr1, 6_vr2, 7_M1, 8_M2
    shellData=np.zeros((len(shells),9))
    for ind,shell in enumerate(shells):
        shellData[ind,0]=shell.ind
        if (shell.m==0):
            continue
        shellData[ind,1]=shell.m
        shellData[ind,2]=shell.rho
        lowEdge=edges[shell.lowEdge]
        highEdge=edges[shell.highEdge]
        if (lowEdge.r > highEdge.r):
            lowEdge=edges[shell.highEdge]
            highEdge=edges[shell.lowEdge]
        shellData[ind,3]=lowEdge.r
        shellData[ind,4]=highEdge.r
        shellData[ind,5]=lowEdge.vr
        shellData[ind,6]=highEdge.vr
        shellData[ind,7]=lowEdge.M
        shellData[ind,8]=highEdge.M
    np.savetxt(fName,shellData,header=headerString)
# END ___ outputData() ___________________

def printEdge(whichEdge): #just outputs all edge properties for debugging
    edge=edges[whichEdge]
    print('________Edge properties: ')
    print('ind: ',edge.ind)
    print('r: ',edge.r)
    print('vr: ',edge.vr)
    print('lSq: ',edge.lSq)
    print('M: ',edge.M)
    print('lowShell: ',edge.lowShell)
    print('highShell: ',edge.highShell)
# END ___ printEdge() ___________________

def printShell(whichShell): #as above with shell properties
    shell=shells[whichShell]
    print('___Shell properties: ')
    print('ind: ',shell.ind)
    print('m: ',shell.m)
    print('rho: ',shell.rho)
    print('lowEdge: ',shell.lowEdge)
    print('highEdge: ',shell.highEdge)
# END ___ printShell() ___________________

cpdef void runSim(int nShells,int nPhase,int nEcc,
                  double tMax,double dt,
                  double minR,double maxR,
                  int nOutput,str simName):
    init(nShells,nEcc,nPhase,minR,maxR)
    cdef:
        int nSteps=int(tMax/dt)+1
        int outputSteps=int((nSteps-1)/nOutput) #should ensure n+1 outputs
        int step
        edgeClass edge
        shellClass shell

    print(len(edges),' edges')
    print(len(shells),' shells')

    # The actual simulation
    for step in np.arange(nSteps):
        # Updates shell radii, then the list of radii
        for edge in edges:
            edge.updateR(dt)
        for shell in shells:
            shell.rho=shell.m/( (4*np.pi/3)*np.abs(edges[shell.lowEdge].r**3 - edges[shell.highEdge].r**3) )
        # Finds mass internal to each shell
        findMass()
        for edge in edges:
            edge.updateV(dt)
    
        if (step%outputSteps==0):
            outputData(step,dt,step/outputSteps,nOutput,simName)
            print('time: ',step*dt)