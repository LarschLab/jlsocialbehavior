# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 21:57:04 2016

@author: johannes
"""

from Tkinter import * 

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc


class App(Tk):
    def __init__(self):
        Tk.__init__(self)

        date1 = (2004, 2, 1)
        date2 = (2004, 4, 12)


        mondays = WeekdayLocator(MONDAY)        
        alldays = DayLocator() 
        weekFormatter = DateFormatter('%b %d')  
        dayFormatter = DateFormatter('%d') 

        quotes = quotes_historical_yahoo_ohlc('INTC', date1, date2)
        if len(quotes) == 0:
            raise SystemExit

        fig = Figure(figsize = (7.5, 4.5), dpi = 100)
        ax = fig.add_subplot(111)


        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
        ax.xaxis.set_major_formatter(weekFormatter)
        candlestick_ohlc(ax, quotes, width=0.6)
        ax.xaxis_date()
        ax.autoscale_view()
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side = TOP,  fill = BOTH,  expand = False)


if __name__ == "__main__":
    app = App()
    app.geometry("800x600+51+51")
    app.title("Candlestick")
    app.mainloop()