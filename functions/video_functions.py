# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:34:22 2016

@author: jlarsch
"""

import numpy as np
import os
import functions.gui_circle as gc
import functions.ImageProcessor as ImageProcessor
import models.geometry as geometry
import cv2
import pandas as pd
import glob
import sys
import subprocess
import tempfile
import shutil

from functions.getMedVideo import getMedVideo


def get_pixel_scaling(aviPath,forceCorrectPixelScaling=0,forceInput=0,bg_file=''):
    #pixel scaling file will typically reside in parent directory where the raw video file lives
    #forceCorrectPixelScaling=0 - force user iput if no previous data exists
    #forceInput=0 - force user input even if data exists - overwrite
    head, tail = os.path.split(aviPath)
    head=os.path.normpath(head)
    try:
        bg_file=glob.glob(head+'\\dishImage*.jpg')[0]
    except:
        #print 'no background image found for pixel scaling, regenerating...'
        bg_file= getMedVideo(aviPath)[1]
    
    parentDir = os.path.dirname(head)
    scaleFile = os.path.join(parentDir,'bgMed_scale.csv')
    
    if np.equal(~os.path.isfile(scaleFile),-2) or forceCorrectPixelScaling:
        #aviPath = tkFileDialog.askopenfilename(initialdir=parentDir,title='select video to generate median for scale information')
        bg_file= getMedVideo(aviPath, bg_file=bg_file)[1]
#        print bg_file, 'run circleGUI'
        scaleData=gc.get_circle_rois(bg_file,'_scale',forceInput)[0]        
      
    elif forceInput or (np.equal(~os.path.isfile(scaleFile),-1) and  forceCorrectPixelScaling):
        scaleData=np.array(np.loadtxt(scaleFile, skiprows=1,dtype=float))
    else:
        print('no PixelScaling found, using 8 pxPmm')
        return 8

    pxPmm=2*scaleData['circle radius']/scaleData['arena size']
    return pxPmm.values[0]


def extract_frames_ffmpeg(aviPath, frames, fps=30):
    """
    Extract specific frames from video using FFmpeg (much faster than OpenCV).
    
    Parameters:
    -----------
    aviPath : str
        Path to video file
    frames : np.array
        Array of frame numbers to extract (should be sorted!)
    fps : float
        Frames per second of the video
        
    Returns:
    --------
    temp_dir : str
        Path to temporary directory containing extracted frames
    frame_files : list
        List of paths to extracted frame files (ordered)
    valid_frames : np.array
        Frame numbers that were actually requested from FFmpeg
    """
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix='ffmpeg_frames_')
    
    total_frames = None
    try:
        # Get video properties to determine fps
        from functions.getVideoProperties import getVideoProperties
        video_info = getVideoProperties(aviPath)
        fps = float(video_info['fps'])
        if 'nb_frames' in video_info and str(video_info['nb_frames']).isdigit():
            total_frames = int(video_info['nb_frames'])
        elif 'duration' in video_info:
            total_frames = int(float(video_info['duration']) * fps)
    except:
        print(f'Could not get fps from video, using default: {fps}')
    
    frame_files = []
    
    # Extract frames using FFmpeg's select filter for batch extraction
    # This is MUCH faster than extracting individual frames
    print(f'Extracting {len(frames)} frames using FFmpeg...')
    
    # Limit frames to video length if we know it
    valid_frames = np.array(frames, dtype=int)
    if total_frames is not None:
        valid_frames = valid_frames[valid_frames < total_frames]
        if valid_frames.shape[0] == 0:
            print('No valid frames within video length. Skipping FFmpeg extraction.')
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, None
        if valid_frames.shape[0] < len(frames):
            print(f'Warning: {len(frames) - valid_frames.shape[0]} frames exceed video length and will be skipped.')

    # Build select expression for FFmpeg (e.g., "eq(n\,100)+eq(n\,200)+eq(n\,300)")
    select_expr = '+'.join([f'eq(n\\,{int(f)})' for f in valid_frames])
    
    output_pattern = os.path.join(temp_dir, 'frame_%04d.png')
    
    cmd = [
        'ffmpeg',
        '-i', aviPath,
        '-vf', f'select={select_expr}',
        '-vsync', '0',  # Don't duplicate frames
        '-q:v', '2',    # High quality
        output_pattern,
        '-loglevel', 'error'  # Only show errors
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # Get list of extracted files in order
        frame_files = sorted(glob.glob(os.path.join(temp_dir, 'frame_*.png')))
        print(f'Successfully extracted {len(frame_files)} frames to {temp_dir}')
    except subprocess.CalledProcessError as e:
        print(f'FFmpeg extraction failed: {e.stderr.decode()}')
        # Clean up and return None to signal fallback to OpenCV
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None
    except FileNotFoundError:
        print('FFmpeg not found. Please install FFmpeg or it will fallback to OpenCV.')
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None
    
    return temp_dir, frame_files, valid_frames


def getAnimalLength(aviPath,frames,coordinates,boxSize=200,threshold=20,invert=False,use_ffmpeg=True):
    """
    Extract animal length from video frames.
    
    Parameters:
    -----------
    use_ffmpeg : bool
        If True, use FFmpeg for frame extraction (faster). Falls back to OpenCV if FFmpeg fails.
    """
    
    eAll = np.zeros((frames.shape[0], coordinates.shape[1], 6))
    temp_dir = None
    frame_images = []
    
    # Try FFmpeg extraction first if requested
    if use_ffmpeg:
        temp_dir, frame_files, valid_frames = extract_frames_ffmpeg(aviPath, frames)

        if frame_files is not None and len(frame_files) > 0:
            print('Using FFmpeg for frame extraction')
            # Map extracted frames to their frame numbers
            frame_map = {}
            for frame_num, frame_file in zip(valid_frames, frame_files):
                img = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    img = cv2.imread(frame_file)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frame_map[int(frame_num)] = img
        else:
            print('FFmpeg extraction failed, falling back to OpenCV')
            use_ffmpeg = False
    
    # Fallback to OpenCV if FFmpeg disabled or failed
    if not use_ffmpeg or len(frame_images) == 0:
        print('Using OpenCV for frame extraction')
        cap = cv2.VideoCapture(aviPath)
    
    # Process frames
    for i in range(frames.shape[0]):
        string = str(i) + ' out of ' + str(frames.shape[0]) + ' frames.'
        sys.stdout.write('\r' + string)
        
        # Get frame image
        if use_ffmpeg and frame_files is not None:
            gray = frame_map.get(int(frames[i]))
        else:
            f = frames[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, image = cap.read()
            if not ret or image is None:
                gray = None
            else:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                except:
                    gray = image

        if gray is None:
            eAll[i, :, :] = np.nan
            continue
        
        # Process each animal in the frame
        for j in range(coordinates.shape[1]):
            g = gray.copy()
            if np.isnan(coordinates[i, j, 0]):
                eAll[i, j, :] = np.nan
            else:
                currCenter = geometry.Vector(*coordinates[i, j, :].astype('int'))
                crop = ImageProcessor.crop_zero_pad(g, currCenter, boxSize)
                
                if invert:
                    crop = 255 - crop
                img_binary = ImageProcessor.to_binary(crop.copy(), threshold, invertMe=False)
                
                contours, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_center = geometry.Vector(crop.shape[0] / 2, crop.shape[1] / 2)
                
                cnt = ImageProcessor.get_contour_containing_point(contours, img_center)
                try:
                    (x, y), (MA, ma), ori = cv2.minAreaRect(cnt[0])
                    eAll[i, j, 0:5] = [x, y, MA, ma, ori]
                    eAll[i, j, 0:2] = eAll[i, j, 0:2] + currCenter
                    mask = crop.copy() * 0
                    cv2.drawContours(mask, cnt[0], -1, (255), -1)
                    eAll[i, j, 5] = cv2.mean(crop, mask=mask)[0]
                except:
                    eAll[i, j, :] = np.nan
    
    # Cleanup
    if temp_dir is not None:
        try:
            shutil.rmtree(temp_dir)
            print(f'\nCleaned up temporary frames from {temp_dir}')
        except:
            print(f'\nWarning: Could not clean up {temp_dir}')
    
    if not use_ffmpeg or len(frame_images) == 0:
        cap.release()
    
    return eAll      


def getAnimalSize(experiment,needFrames=2000,numFrames=40000,boxSize=200,e2=[]):
    avi_path=experiment.expInfo.aviPath
    head, tail = os.path.split(avi_path)
    sizeFile = os.path.join(head,'animalSize.txt')    
    

    
    if ~np.equal(~os.path.isfile(sizeFile),-2):
        print('determining animalSize from data')
        haveFrames=0
        frames=np.zeros(needFrames).astype('int')
        dist=np.zeros(needFrames)
        
        triedFr=[]
        triedD=[]
        while haveFrames<needFrames:
            tryFrame=np.random.randint(1000,numFrames,1)
            minDist=np.max(np.abs(np.diff(experiment.rawTra[tryFrame,:,:],axis=1)))
            if minDist>boxSize:
                frames[haveFrames]=int(tryFrame)
                dist[haveFrames]=minDist
                haveFrames += 1
            else:
                triedFr.append(tryFrame)
                triedD.append(minDist)
        
        
        tra=experiment.rawTra[frames,:,:]
        if e2!=[]:
            tra[:,0,:]=experiment.rawTra[frames,1,:]

            tra[:,1,:]=e2.rawTra[frames,1,:]
            tra[:,0,0]=tra[:,0,0]+512
            #tra[:,:,1]=512-tra[:,:,1]
            print('using shifted secondAnimal trajectory')
        
        #if (int(experiment.expInfo.videoDims[0])/float(tra.max()))>2:
            
        
        tmp=getAnimalLength(avi_path,frames,tra)
        brightness=np.mean(tmp[:,:,5],axis=0).astype('int')
        MA=np.max(tmp[:,:,2:4],axis=2)
        bins=np.linspace(0,100,101)
        anSize=[np.argmax(np.histogram(MA[:,0],bins=bins)[0]),np.argmax(np.histogram(MA[:,1],bins=bins)[0])]
    
        df=pd.DataFrame({'anID':[1,2],'anSize':anSize,'brightness':brightness},index=None)
        df.to_csv(sizeFile,sep='\t',index=False)
        anID=np.array([1,2])

        ret=np.vstack([anID,anSize]).T     
    else:
#        print 'loading saved animalSize'
        tmp = pd.read_csv(sizeFile, dtype=int, delim_whitespace=True, skipinitialspace=True)
        
        ret = np.array(tmp[[0, 1]].values)  # np.array(np.loadtxt(sizeFile, skiprows=1,dtype=int))
        
    return ret
    
