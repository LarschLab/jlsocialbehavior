# -*- coding: utf-8 -*-
"""
Created on Sat Aug 06 22:51:16 2016

@author: johannes
"""

def getSublists(L,n):
    List=L
    sublists=[]
    for i in range(len(L)-(n-1)):
        print ['i: ', i]
        ii=0
        sub=[]
        while ii<= n-1:
            print ['ii: ', ii]
            a=List[ii+i]
            sub.append(a)
            ii+=1
        sublists.append(sub)
    
    return sublists