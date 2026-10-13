# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 11:03:00 2016

@author: jlarsch
"""


from psychopy import visual, core  # import some libraries from PsychoPy

#create a window
mywin = visual.Window([800,600], monitor="testMonitor", units="deg")
#create some stimuli
grating = visual.GratingStim(win=mywin, mask="circle", size=3, pos=[-4,0], sf=3)

#draw the stimuli and update the window
for frameN in range(200):
    grating.setPhase(0.05, '+') # advance phase by 0.05 of a cycle
    grating.draw()
    mywin.update()


#pause, so you get a chance to see it!
#core.wait(5.0)