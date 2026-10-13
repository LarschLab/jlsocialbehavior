# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:56:15 2016

@author: jlarsch
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import models.geometry as geometry
import functions.peakdet as peakdet

def processing_video(asp,experiment,path,frames,start=0):
    
    #read video frame
    outSize=300
    video = cv2.VideoCapture(path)
    img_frame_first = video.read()[1]
    height_ori, width_ori = img_frame_first.shape[:2]
    resize_factor=height_ori/float(outSize)
    
    frameList=np.arange(start,start+frames,1)
    fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    fps=30
    videoOut1 = cv2.VideoWriter('e:/VidProc2.avi',fourcc,fps,(3*outSize ,3*outSize),True)
    fig = plt.figure(figsize=(6, 1), dpi=100,facecolor='w', edgecolor='w')
    plt.plot(1)
    plt.tight_layout()
    try:
        buffSize=50
        posBuffer=np.zeros((buffSize,2,2))
        buff=0
        
        for i in frameList:
            cont=np.ones((3 * outSize,3*outSize,3),dtype='uint8')*255
            video.set(cv2.CAP_PROP_POS_FRAMES,i)
            img_frame_original = video.read()[1]
            img_frame_original=adjust_gamma(img_frame_original,0.5)
            pos=np.array([asp[0].trajectory[i,:],asp[1].trajectory[i,:]])
            add_surround=.2 #fraction increase beyond fish
            add_surround_min=150
            s=img_frame_original.shape[0]
            cropSize=np.max(np.diff(pos,axis=0))
            surround=np.max([add_surround_min,cropSize*add_surround])
            cropSize=int(cropSize+surround)
            cropSize=np.max([2*outSize,cropSize])
            pairCenter=np.mean(pos,axis=0).astype('int')
            xmin=int(np.max([0,pairCenter[0]-cropSize/2]))
            xmax=int(np.min([s,pairCenter[0]+cropSize/2]))
            ymin=int(np.max([0,pairCenter[1]-cropSize/2]))
            ymax=int(np.min([s,pairCenter[1]+cropSize/2]))


            img_pos=img_frame_original.copy()
            
            for a in range(2):
                anCol=(255*(a==0),(a==1)*255,0)
                if np.mod(i,1)==0:
                    pos=asp[a].trajectory[i,:]
                    posBuffer[0,a,:]=pos
                    buff +=1
                         
                for b in np.arange(0,np.min([buff,buffSize]),1):
    
                    opa=(1-float(b)/float(buffSize))/10
                    img_pos=draw_alpha(img_pos,tuple(posBuffer[b,a,:].astype('int')),10,anCol,2,opa)
                                
                
                #cv2.circle(img_pos, tuple(pos.astype('int')), 50, (255,255,255), -1)
                #cv2.circle(img_pos, tuple(pos.astype('int')), 65, (0,0,0), 2)
               # cv2.addWeighted(img_pos, 0.5, img_frame_original, 0.5, 0, img_frame_original)
                
                curr_ori=asp[a].fish_orientation_elipse[i]  
                if not np.isnan(curr_ori):
                    curr_ori=np.mod(180-curr_ori,360)+90
                    oriVec=geometry.Vector.new_vector_from_angle(curr_ori)*30
                    oriVec.draw(img_frame_original,geometry.Vector(*posBuffer[0,a,:]),anCol)
                
                crop=asp[a].frAll_rot[i,100:200,50:250]
                cropCol=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)
                cropCol=255-cropCol
                cropCol[255-crop<10,:]=anCol
                cont[outSize+a*100:outSize+a*100+cropCol.shape[0],:cropCol.shape[1],:]=cropCol

            overlay=img_pos.copy()
            gridColor=[200,200,200]
            overlay[:,::50,:]=gridColor
            overlay[::50,:,:]=gridColor
            
            cv2.line(overlay,tuple(posBuffer[0,0,:].astype('int')),tuple(posBuffer[0,1,:].astype('int')),(0,0,0))
            arenaCenter=tuple(experiment.expInfo.arenaCenterPx.astype('int'))
            arenaDiameter=np.mean(experiment.maxPixel)-np.mean(experiment.minPixel)
            cv2.circle(overlay, arenaCenter, int(arenaDiameter/2), (0,0,0), 10)            
            
            overlay[img_frame_original<200]=img_frame_original[img_frame_original<200]
            img=cv2.resize(overlay,(outSize,outSize),interpolation = cv2.INTER_AREA)
            
            cont[:img.shape[0],:img.shape[0],:]=img

            img_pairCrop=overlay[ymin:ymax,xmin:xmax]
            img_pairCrop=cv2.resize(img_pairCrop,(2*outSize,2*outSize),interpolation = cv2.INTER_AREA)
            cont[:img_pairCrop.shape[0],outSize:,:]=img_pairCrop
            
            tail1=np.nanmean(asp[0].allDist.T,axis=0)
            bout_start1=peakdet.detect_peaks(tail1,1,8)
            tail1=tail1/np.nanmax(tail1)
            tail2=np.nanmean(asp[1].allDist.T,axis=0)
            bout_start2=peakdet.detect_peaks(tail2,1,8)
            tail2=tail2/np.nanmax(tail2)
            tail=np.array([tail1,tail2],ndmin=2).T
            bout_start=np.array([bout_start1,bout_start2])
            tailFrame=video_plot(fig,tail,i,bout_start)
            
            err1=asp[0].deviation
            err2=asp[1].deviation
            err=np.array([err1,err2],ndmin=2).T
            errFrame=video_plot(fig,err,i)
            
            cont[img_pairCrop.shape[0]:img_pairCrop.shape[0]+100,outSize:,:]=tailFrame
            cont[img_pairCrop.shape[0]+100:img_pairCrop.shape[0]+200,outSize:,:]=errFrame
            
            videoOut1.write(cont)
            posBuffer=np.roll(posBuffer,1,axis=0)
    
        videoOut1.release()
            
        
    except Exception as e: 
        print(e)
        videoOut1.release()
    plt.close(fig)
    
def draw_alpha(img,center,rad,col,thick,opacity):
    
    overlay = img.copy()
    # (2) draw shapes:
    cv2.circle(overlay, center, rad, col, thick)
    # (3) blend with the original:
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
    return img
    
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
 
def video_plot(fig,data,t,events=None,x_ax_l=100):
    
    #ax = fig.add_subplot(111)
    plt.cla()
    plt.xlim(0,x_ax_l/30)
    plt.ylim(np.nanmin(data)-.1,np.nanmax(data)+.2)

    # set initial data points and draw them on the figure
    x = ((np.arange(t+1)+1)/30.0)
    d = data[:t+1,:]
    #print x.shape
    #print d.shape
    plt.plot(x, d)
    ymin, ymax = plt.ylim()
    #plot 'events'

    if events!=None:
        for a in range(events.shape[0]):
            anCol=(0,(a==1)*1,1*(a==0))
            currList=events[a]
            currList=currList[currList<t]
            currList=currList[currList>t-x_ax_l]
            for j in range(currList.shape[0]):
                plt.plot([currList[j]/30.0,currList[j]/30.0], [ymin, ymax], ':', color=anCol, lw=1)
                plt.plot(currList[j]/30.0,1-a*0.1,'o',color=anCol)
        

    #   change the x limits of the graph if needed
    if x[-1] > (x_ax_l/30):
        plt.xlim(x[-1]-(x_ax_l/30), x[-1])
    
    plt.plot([x[-1],x[-1]], [ymin, ymax], 'k', lw=1)

    
    frame1 = plt.gca()
    frame1.axes.yaxis.set_ticklabels([])
    frame1.spines['top'].set_visible(False)
    frame1.spines['right'].set_visible(False)
    frame1.spines['bottom'].set_visible(False)
    frame1.spines['left'].set_visible(False)
    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left='off',      # ticks along the bottom edge are off
    right='off')         # ticks along the top edge are off
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    top='off',      # ticks along the bottom edge are off
    bottom='on')         # ticks along the top edge are off
    
    fig.canvas.draw()
    out = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    out = out.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    out=cv2.cvtColor(out,cv2.COLOR_RGB2BGR)


    return out
    
processing_video(asp,experiment,af,48,0)

