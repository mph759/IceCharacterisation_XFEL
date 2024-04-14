# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:38:44 2024

@author: S3599678
"""

import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from paltools import run, energy2wavelength, radial_average
from ice_peak_predict import IcePeakPrediction
from peak_fitting import HexIceFitting  

class NewFileHandler(FileSystemEventHandler):
    def __init__(self, reset_inactivity_counter_callback, func=None, *arg, **kwargs):
        self.reset_inactivity_counter_callback = reset_inactivity_counter_callback
        self.func = func
        self.arg = arg
        self.kwargs = kwargs

    def on_created(self, event):
        print(f'New file created: {event.src_path}')
        if event.src_path.endswith('.tiff'):
            # Reset the inactivity counter whenever a new file is created
            self.reset_inactivity_counter_callback()
            # Process each new file in a separate thread
            if self.func is not None:
                thread = threading.Thread(target=self.func(*self.args, **self.kwargs), args=(event.src_path,))
                thread.start()

def watch_folder(data_folder, interval, max_intervals_without_files, func, *arg, **kwargs):
    inactivity_counter = 0

    def reset_inactivity_counter():
        nonlocal inactivity_counter
        inactivity_counter = 0

    observer = Observer()
    event_handler = NewFileHandler(reset_inactivity_counter)
    observer.schedule(event_handler, path=data_folder, recursive=False)

    observer.start()
    try:
        while inactivity_counter < max_intervals_without_files:
            time.sleep(interval)
            inactivity_counter += 1
            print(f"Inactivity counter: {inactivity_counter}")
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        observer.stop()
        observer.join()
        print("No new files recorded for a while, script stopped.")

def process_new_file(self, file_path):
     time.sleep(1)
     wavelength = energy2wavelength(15e3)
     amplitude = [6.5e-5, 1.3e-4, 2.9e-5]
     mean = [1.6, 1.7, 1.8]
     stddev = [0.01, 0.01, 0.01]
     current_array = np.array(Image.open(file_path))
     r, image = radial_average(current_array, 20)
     np.save(out_path, (r,image))
     print("done")
     #plt.plot(r, image)
     #plt.savefig(plot_path + "_plot.png")
     #plt.close()

def foo():
    pass

if __name__ == '__main__':
    path_to_watch = "C://Users//s3599678//OneDrive - RMIT University//PhD//misc//Seoul//scripts//IceCharacterisation_XFEL-master//test//powder//"
    out_path = path_to_watch+'//plots//'
    watch_folder(path_to_watch, 1, 2, func=foo)