# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:52:06 2016

@author: jlarsch
"""

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt



class plotClass(object):
    def __init__(self):
        self.PdfFile='c:/data/test.pdf'
        self.pdf = PdfPages(self.PdfFile)
        self.foo1()
        self.foo2()
        self.pdf.close()        
        
    def foo1(self):
        plt.bar(1,1)
        self.pdf.savefig()
    
    def foo2(self):
        plt.bar(1,2)
        self.pdf.savefig()
        
            
test=plotClass()