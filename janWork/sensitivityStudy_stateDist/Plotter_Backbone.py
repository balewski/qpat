import os
import numpy as np
#from matplotlib import cm as cmap
#from matplotlib.ticker import MaxNLocator


import socket  # for hostname
import time

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

#............................
#............................
#............................
class Plotter_Backbone(object):
    def __init__(self, args):
        if args.noXterm:
            print('disable Xterm')
            import matplotlib as mpl
            mpl.use('Agg')  # to plot w/o X-server
        import matplotlib.pyplot as plt
        print(self.__class__.__name__,':','Graphics started')
        plt.close('all')
        self.plt=plt
        self.figL=[]
        self.outPath=args.outPath
        for xx in [ self.outPath]:
            if os.path.exists(xx): continue
            print('Aborting on start, missing  dir:',xx)
            exit(99)


    #............................
    def display_all(self,args,ext,pdf=1):
        if len(self.figL)<=0: 
            print('display_all - nothing top plot, quit')
            return
        if pdf:
            for fid in self.figL:
                self.plt.figure(fid)
                self.plt.tight_layout()                
                figName='%s/%s_%s_f%d.png'%(self.outPath,args.prjName,ext,fid)
                print('Graphics saving to %s  ...'%figName)
                self.plt.savefig(figName)
                if pdf==2:
                    self.plt.savefig(figName+'.eps')
                    self.plt.savefig(figName+'.png')
                    print(' also save .eps & .png')
        self.plt.show()

# figId=self.smart_append(figId)
#...!...!....................
    def smart_append(self,id): # increment id if re-used
        while id in self.figL: id+=1
        self.figL.append(id)
        return id

