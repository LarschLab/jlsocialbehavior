# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 13:36:42 2015

@author: jlarsch
"""

import wx

def ask(parent=None, message='', default_value=''):
    dlg = wx.TextEntryDialog(parent, message, defaultValue=default_value)
    dlg.ShowModal()
    result = dlg.GetValue()
    dlg.Destroy()
    return result

# Initialize wx App
app = wx.App()
app.MainLoop()

# Call Dialog
x = ask(message = 'What is your name?')
print 'Your name was', x