# conda activate SI_fMRI
# conda activate SI_fMRI
import moten
import numpy as np
import os
import cv2
import pickle
from multiprocessing import Pool

path = '../test_media_middle10s'
motion_features = {}

def process_video(file):
    try:
        full_path = os.path.join(path, file)
        cam = cv2.VideoCapture(full_path)
        
        # Gather video properties
        fps = cam.get(cv2.CAP_PROP_FPS)
        frame_count = cam.get(cv2.CAP_PROP_FRAME_COUNT)
        
        print(f"Processing {file}: FPS={fps}, Frame Count={frame_count}")
        
        # Stream and convert the RGB video into a sequence of luminance images
        luminance_images = moten.io.video2luminance(full_path)

        # Create a pyramid of spatio-temporal gabor filters
        nimages, vdim, hdim = luminance_images.shape
        pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=20)

        # Compute motion energy features
        moten_features = pyramid.project_stimulus(luminance_images)
        
        return file, moten_features
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return file, None

if __name__ == '__main__':
    files = [file for file in os.listdir(path) if file.endswith(".mp4")]
    
    with Pool() as pool:
        results = pool.map(process_video, files)
        
    motion_features = {file: features for file, features in results if features is not None}
    
    with open('motion_energies_test_middle10s', "wb") as f:
        pickle.dump(motion_features, f)