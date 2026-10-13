# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:17:29 2016

@author: jlarsch
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
fn='http://cordis.europa.eu/data/cordis-h2020projects.xlsx'

#df=pd.read_excel(fn)

mask=df['topics'].map(lambda x: x.startswith('MSCA'))
grouped=df[mask].groupby('fundingScheme')
total=grouped['totalCost'].aggregate(np.sum)
total.plot(kind='bar')
grouped.size()



mask=df['fundingScheme'].map(lambda x: x.startswith('MSCA-IF-EF'))
grouped=df[mask].groupby('coordinator')
actionsPerInstitute=df.groupby('coordinator')['coordinator'].count().sort_values(ascending=False)

fellowsPerInstitute=df[mask].groupby('coordinator')['coordinator'].count().sort_values(ascending=False)
total=grouped['totalCost'].aggregate(np.sum)
total=total.sort_values(ascending=False)
total.plot(kind='bar')
grouped.size()

sort1 = df[mask].ix[grouped[['totalCost']].transform(sum).sort_values('totalCost').index]
sel=df['participantCountries']

eg=pd.DataFrame({'participants': ['UK;SE;DE;FR;NL','NL;ES;IT;UK','DE;ES;UK;FR;EL','NL;IL;UK;IT;ES','DE;BE;EL;FR;UK;IT','DE;S']})
              

import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

nodes=set()
G = nx.Graph()

selClean=sel[sel.notnull()].values


#for index, row in eg.iterrows():
for row in selClean:
    countries=row.split(';')
    #countries=row.participants.split(';')
    #print countries
    for c in countries:
        if c not in nodes:
            G.add_node(c)
            nodes.add(c)
            
    edges = combinations(countries, 2)
    
    for a,b in edges:
        #print a,b
        if G.has_edge(a,b):
            #print 'found edge'
            G[a][b]['weight'] += .001
        else:
            G.add_edge(a,b)
            G[a][b]['weight'] = 0

edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]

#nx.draw(G, edges=edges, width=weights)
pos = nx.circular_layout(G)
nx.draw_networkx(G,edges=edges,width=weights)