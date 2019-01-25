# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:22:22 2016

@author: jlarsch

-This class contains 'fundamental' and 'derived' time series belonging to a specific animal

-'fundamental' time series are extracted directly from the video
    -fundamentals are stored in the parent experiment class for all animal siblings
    -fundamentals get loaded into memory from csv files
-'derived' time series are transformed fundamentals, e.g. speed, angles, heading
    -derived time series are computed during run time at each call

-this class has one parent animal. 
-Using class composition to link the parent animal into this class (self.animal)
-for derived time series related to neighbor animal, self.animal is the 'focal' animal
-the neighbor can be found in self.animal.neighbor
-each animal has one class of this type

"""
import numpy as np
from models.geometry import Trajectory
import functions.matrixUtilities_joh as mu
import scipy.stats as sta
import models.geometry as geometry
import functions.peakdet as pkd

class AnimalTimeSeriesCollection:
    def __init__(self):
        self.animalIndex = None
        self.mapBins = np.arange(-31, 32)
        self.pxPmm = None
        self.fps = None
    
    # Call this function from an animal class to link it with this class
    def linkAnimal(self, animal):
        self.animal = animal
        self.ID = animal.ID
        if self.animal.paired:
            self.animalIndex = self.animal.pair.animalIDs[self.ID]
            self.pxPmm = self.animal.pair.experiment.expInfo.pxPmm
            self.fps = self.animal.pair.experiment.expInfo.fps
        else:
            self.animalIndex = self.ID
            self.pxPmm = self.animal.experiment.expInfo.pxPmm
            self.fps = self.animal.experiment.expInfo.fps
        animal.add_TimeSeriesCollection(self)
    
    # function to shift fundamental time series upon loading to generate control data
    def timeShift(self, x):
        return np.roll(x, self.animal.pair.shift[self.animal.ID], axis=0)

# --------------------------
# fundamental time series
# --------------------------


    def rawTra(self):
        a = self.animalIndex
        currCols = [a * 3, a * 3 + 1]
        if self.animal.paired:
            rng = self.animal.pair.rng
            x = self.animal.pair.experiment.rawTra.iloc[rng[0]:rng[1], currCols]
            return Trajectory(self.timeShift(x))
        else:
            x = self.animal.experiment.rawTra.iloc[:, currCols]
            return Trajectory(np.roll(x, 0, axis=0))

    # def trackedHeading(self):
    #     a = self.animalIndex
    #     rng = self.animal.pair.rng
    #     currCols = [a * 3 + 2]
    #     x = self.animal.pair.experiment.rawTra.iloc[rng[0]:rng[1], currCols]
    #     return self.timeShift(x)
        
# --------------------------
# derived time series
# --------------------------
        
    # convert pixels to mm and center on (0,0)
    def position(self):
        # currCenterPx = self.animal.pair.experiment.expInfo.rois[self.animalIndex, -1] + 2
        currCenterPx = 0
        arenaCenterPx = np.array([currCenterPx, currCenterPx])
        x = (self.rawTra().xy-arenaCenterPx) / self.pxPmm
        return Trajectory(x)
        
    def position_smooth(self):
        x = self.position().xy
        return Trajectory(mu.smooth(x, window_len=5, window='hamming')[:9000, :])
        
    def positionPol(self):
        x = [mu.cart2pol(*self.position().xy.T)]
        x = np.squeeze(np.array(x).T)
        return Trajectory(x)
        
    def d_position_Smooth(self):
        x = np.diff(self.position_smooth().xy, axis=0)
        return Trajectory(x)
        
    def d_position(self):
        x = np.diff(self.position().xy, axis=0)
        return Trajectory(x)
        
    def dd_position(self):
        x = np.diff(self.d_position_Smooth().xy, axis=0)
        return Trajectory(x)
        
    def travel_smooth(self):
        x = mu.distance(*self.d_position_Smooth().xy.T)
        return x
        
    def travel(self):
        x = mu.distance(*self.d_position().xy.T)
        return x
        
    def speed(self):
        return self.travel() * self.fps
    
    def speed_smooth(self):
        return self.travel_smooth() * self.fps
    
    def totalTravel(self):
        return np.nansum(np.abs(self.travel()))
        
    def accel(self):
        return np.diff(self.speed_smooth())
        
    def heading(self):
        return mu.cart2pol(*self.d_position_Smooth().xy.T)[0] #heading[0] = heading, heading[1] = speed
        #heading: pointing right = 0, pointung up = pi/2  

    def d_heading(self):
        return np.diff(self.heading())
        
    #currently, this is the position of the neighbor, relative to focal animal name is misleading...
    def position_relative_to_neighbor(self):
        x = self.animal.neighbor.ts.position_smooth().xy - self.position_smooth().xy
        return Trajectory(x)
        
    #rotate self to face up in order to map neighbor position relative to self
    def position_relative_to_neighbor_rot(self):
        relPosPol = [mu.cart2pol(*self.position_relative_to_neighbor().xy.T)]
        relPosPolRot = np.squeeze(np.array(relPosPol).T)[:-1, :]
        relPosPolRot[:, 0] = relPosPolRot[:, 0]-self.heading()
        x = [mu.pol2cart(relPosPolRot[:, 0], relPosPolRot[:, 1])]
        x = np.squeeze(np.array(x).T)
        return Trajectory(x)
    
    #acceleration using rotation corrected data
    #effectively splits acceleration into speeding [0] and turning [1]          
    def dd_pos_pol(self):
        x = [mu.cart2pol(*self.dd_position().xy.T)]
        x = np.squeeze(np.array(x)).T
        return Trajectory(x)
        
    def dd_pos_pol_rot(self):
        x_rot = self.dd_pos_pol().xy
        x_rot[:, 0] = x_rot[:, 0]-self.heading()[:-1]
        x_rot_cart = [mu.pol2cart(x_rot[:, 0], x_rot[:, 1])]
        x_rot_cart = np.squeeze(np.array(x_rot_cart)).T
        return Trajectory(x_rot_cart)
    
    
    #creates the neighbormat for current animal (where neighbor was)
    #this map seems flipped both horizontally and vertically! vertical is corrected at plotting.    
    def neighborMat(self):
        mapBins = self.mapBins
        neighborMat = np.zeros([62, 62])
        pos = self.position_relative_to_neighbor_rot().xy
        pos = pos[~np.isnan(pos).any(axis=1), :]
        neighborMat = np.histogramdd(pos,
                                     bins=[mapBins, mapBins],
                                     normed=True)[0]*neighborMat.shape[0]**2
        return neighborMat
        
#-------simple bout analysis------
    def boutStart(self):
        return pkd.detect_peaks(self.speed_smooth(), mph=5, mpd=8)
    
#------Force matrices-------
#creates force matrix (how did focal animal accelerate depending on neighbor position)
        
    #speedAndTurn - using total acceleration
    def ForceMat_speedAndTurn(self):
        mapBins = self.mapBins
        position_relative_to_neighbor_rot = self.position_relative_to_neighbor_rot()
        ForceMat = sta.binned_statistic_2d(position_relative_to_neighbor_rot.x()[1:],
                                           position_relative_to_neighbor_rot.y()[1:],
                                           self.accel(),
                                           bins=[mapBins, mapBins])[0]
        return ForceMat
        
    #speed - using only acceleration component aligned with heading
    def ForceMat_speed(self):
        mapBins = self.mapBins
        position_relative_to_neighbor_rot = self.position_relative_to_neighbor_rot()
        return sta.binned_statistic_2d(position_relative_to_neighbor_rot.x()[1:],
                                       position_relative_to_neighbor_rot.y()[1:],
                                       self.dd_pos_pol_rot().xy[:, 0],
                                       bins=[mapBins, mapBins])[0]
    
    #turn - using only acceleration component perpendicular to heading   
    def ForceMat_turn(self):
        mapBins = self.mapBins
        position_relative_to_neighbor_rot = self.position_relative_to_neighbor_rot()
        return sta.binned_statistic_2d(position_relative_to_neighbor_rot.x()[1:],
                                       position_relative_to_neighbor_rot.y()[1:],
                                       self.dd_pos_pol_rot().xy[:, 1],
                                       bins=[mapBins, mapBins])[0]
    
    #percentage of time the neighbor animal was in front vs. behind focal animal
    def FrontnessIndex(self):
        PosMat = self.neighborMat()
        front = np.nansum(np.nansum(PosMat[:31, :], axis=1), axis=0)
        back = np.nansum(np.nansum(PosMat[32:, :], axis=1), axis=0)
        return (front-back)/(front+back)
    
    #angle connecting self to neighbor animal    
    def rangeVector(self):
        p1 = self.centroidContour().xy
        p2 = self.animal.neighbor.ts.centroidContour().xy
        angle_centroid_connect = geometry.get_angle_list(p1, p2)
        x = np.mod(180-angle_centroid_connect, 360)
        return x
    
    #difference between heading and range vector
    def errorAngle(self):
        x=geometry.smallest_angle_difference_degrees(self.fish_orientation_elipse(),self.rangeVector())
        return x
    
    #generate a histogram of position over disctance from arena center
    #used in a hacky way to determine stacking order of the two animals in experiments
    #where two dishes are stacked on top of each other.
    def PolhistBins(self):
        #maxOut=self.animal.pair.max_out_venture()
        maxOut = self.animal.pair.experiment.expInfo.arenaDiameter_mm/2.
        x = np.linspace(0, maxOut, 100)
        return x
        
    def Pol_n(self):
        histData = self.positionPol().y()
        histData = histData[~np.isnan(histData)]
        bins = self.PolhistBins()
        pph, ppeb = np.histogram(histData, bins=bins, normed=1)
        
        x = (pph/(np.pi*ppeb[1:]*2))*np.pi*((ppeb[-1])**2)
        return x

