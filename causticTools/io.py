import bokeh
import bokeh.plotting as blt
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from . import cythonMisc

def plotDens(fig,whichOutput,nOutput,simName):
    # 0_shellIndex, 1_shellMass, 2_shellDens, 3_r1, 4_r2, 5_vr1, 6_vr2, 7_M1, 8_M2
    shellData=np.genfromtxt('output/'+simName+'.'+str(int(whichOutput))+'_'+str(int(nOutput))+'.txt')
    #nasty little hack to get points just either side of the steps in density
    rads=np.hstack((shellData[:,3],shellData[:,4]))
    rads=np.hstack((rads,rads))
    rads=np.sort(rads)
    dens=cythonMisc.findDens(rads,shellData)
    
    fig=plotDensProfile(fig,rads,dens,minR,maxR,whichOutput/nOutput)
    return fig


def plotSmoothDens(fig,whichOutput,nOutput,simName,smoothLength,minR,maxR):
    # 0_shellIndex, 1_shellMass, 2_shellDens, 3_r1, 4_r2, 5_vr1, 6_vr2, 7_M1, 8_M2
    fName='output/'+simName+'.'+str(int(whichOutput))+'_'+str(int(nOutput))+'.txt'
    print('reading from: ',fName)
    shellData=np.genfromtxt(fName)
    dr=0.25
    nBins=int((maxR-minR)/dr)
    rads=np.linspace(minR+dr,maxR-dr,nBins)
    dens=cythonMisc.findSmoothDens(rads,nBins,shellData,smoothLength)
    
    fig=plotDensProfile(fig,rads,dens,minR,maxR,whichOutput/nOutput)
    return fig
    
def plotDensProfile(fig,Rs,Rhos,minR,maxR,colour): #if alread have radii and densities plot directly
    cols=bokeh.palettes.Viridis11
    fig.line(Rs,Rhos,color=cols[int(11*colour)],line_width=2)
    fig.xaxis.axis_label='r (pc)'
    fig.yaxis.axis_label='rho (Msun pc-3)'
    return fig
    
def findSmoothDens(rads,nBins,shellData,smoothLength):
    return cythonMisc.findSmoothDens(rads,nBins,shellData,smoothLength)
    
def findDens(shellData):
    rads=np.hstack([shellData[:,3],shellData[:,4]]) #all radii
    rads=np.unique(rads) #only unique radii
    rads=np.hstack([rads,rads]) #doubled up (for boxy plot)
    rads=np.sort(rads) #sorted
    return rads,cythonMisc.findDens(rads,shellData)