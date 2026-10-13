# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 09:48:38 2016

@author: jlarsch
"""

import networkx as nx
import matplotlib.pyplot as plt

#generate a graph with weights
a_netw=nx.Graph()
a_netw.add_edge('a','b',weight=6)
a_netw.add_edge('a','c',weight=2)
a_netw.add_edge('c','d',weight=1)
a_netw.add_edge('c','e',weight=7)
a_netw.add_edge('c','f',weight=9)
a_netw.add_edge('a','d',weight=3)

#creating a color list for each edge based on weight

a_netw_edges = a_netw.edges()
a_netw_weights = [a_netw[source][dest]['weight'] for source, dest in a_netw_edges]

#scale weights in range 0-1 before assigning color 
maxWeight=float(max(a_netw_weights))
a_netw_colors = [plt.cm.Blues(weight/maxWeight) for weight in a_netw_weights]


#suppress plotting for the following dummy heatmap
plt.ioff()

#multiply all tuples in color list by scale factor
colors_unscaled=[tuple(map(lambda x: maxWeight*x, y)) for y in a_netw_colors]
#generate a 'dummy' heatmap using the edgeColors as substrate for colormap
heatmap = plt.pcolor(colors_unscaled,cmap=plt.cm.Blues)

#re-enable plotting
plt.ion()

fig,axes = plt.subplots()
nx.draw_networkx(a_netw, edges=a_netw_edges, width=10, edge_color=a_netw_colors, ax=axes)
axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)

#add colorbar
cbar = plt.colorbar(heatmap)
cbar.ax.set_ylabel('edge weight',labelpad=15,rotation=270)

