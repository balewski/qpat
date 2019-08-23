from Plotter_Backbone import Plotter_Backbone

import numpy as np
#............................
#............................
#............................
class Plotter_CircSensitivity(Plotter_Backbone):
    def __init__(self, args):
        Plotter_Backbone.__init__(self,args)

    #............................
    def stateDistance_spectra(self, resultD, metaD, figId=1): 
        posV=resultD['pos']
        numPos=len(posV)
        fidelV=resultD['fidel']
        maxDist=metaD['maxDist']
        gatePosInfo=metaD['gatePosInfo']
        #print('aa',gatePosInfo.keys())
        print('plot_stateDist num pos=%d plots=%d maxDist='%(numPos, len(fidelV)),maxDist)
        if len(posV)<=0:
            print('no data, skip'); return
            
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,4))
        nrow,ncol=1,1
        #  grid is (yN,xN) - y=0 is at the top,  so dumm
        
        # . . . . . . . . . . . . . . . .
        ax = self.plt.subplot(nrow,ncol,1)
        bins = np.linspace(0,maxDist, 100)
        
        for idx,pos in enumerate(posV):
            print(idx,'pos',pos,idx,pos[0])
            posStr=str(pos)
            posLab=''
            for gid in pos:
                if len(posLab)>0: posLab+=' + '
                PI=gatePosInfo[gid]
                posLab+='g%d:%s(%s)'%(gid,PI['name'],str(PI['qubits']))
            if idx==8: posLab='(etc...)'
            if idx>8: posLab=''
            if len(posLab)>60 : posLab=posLab[:60]+' skip more...'
            
            one=fidelV[idx]
            mean=resultD['mean'][idx]

            ax.hist(one, bins, alpha=0.5,histtype='step',label=posLab)

            ax.axvline(mean)
            #ty=(0.3+0.7*(idx/numPos))*yMax
            #ax.text(mean,ty,posStr)

        tit='circ='+metaD['circName']+', sigTheta=%.2f rad'%metaD['sigTheta']
        ax.set(xlabel="State Distance", ylabel='num samples ',title=tit)

        ax.legend(loc='upper right',title='noise source: %d experiments'%numPos)
        ax.grid(True)
        

    #............................
    def stateDistance_average(self, resultD, metaD,tit, figId=2):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,4))
        nrow,ncol=1,1

        # . . . . . . . . . . . . . . . .
        ax = self.plt.subplot(nrow,ncol, 1)
        posV=resultD['pos']
        G=[str(x) for x in posV]; Y=resultD['mean']; eY=resultD['stder']
        #print('G',G)
        #print('Y',Y)
        #print('eY',eY)
        ax.errorbar( G, Y, yerr=eY, fmt='o')
        #ax.set_ylim(min(Y)-0.001, np.max(Y) +1e-3 )

       
        ax.set_xticklabels( G , rotation=80, ha="right")
        ax.set_ylim(0,)
        ax.set_xlabel( "noise source gate(s)" )
        ax.set_ylabel( "State Distance" )
        ax.set_title(  'sigTheta=%.2f rad, numEve=%d, elaTime=%.1f min'%(metaD['sigTheta'],metaD['numEve'],metaD['elaTime']/60.))
        ax.grid()

        print('qqq',len(G))
        # compute weighted average
        wY=np.power(eY,-2)
        print('err=',eY)
        print('w=',wY)

        mean=np.average(Y,weights=wY)
        print('mean=',mean)
        ax.axhline(mean,linestyle='--',color='r')
        ax.text(0.1,0.1,'mean=%.3f'%mean,color='r',size=20,transform=ax.transAxes)
