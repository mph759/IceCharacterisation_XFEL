# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:48:39 2024

@author: S3599678 Stefan Paporakis
"""

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class NewFileHandler(FileSystemEventHandler):
    def __init__(self, new_file_callback):
        self.new_file_callback = new_file_callback

    def on_created(self, event):
        # This function is called when a new file is created
        print(f'New file created: {event.src_path}')
        self.new_file_callback()  # Call the callback to reset the inactivity counter

def watch_folder(data_folder, interval, max_intervals_without_files):
    inactivity_counter = 0

    def reset_inactivity_counter():
        nonlocal inactivity_counter
        inactivity_counter = 0

    event_handler = NewFileHandler(reset_inactivity_counter)
    observer = Observer()
    observer.schedule(event_handler, path=data_folder, recursive=False)
    
    observer.start()
    try:
        while inactivity_counter < max_intervals_without_files:
            time.sleep(interval)
            inactivity_counter += 1
            print(f"no new file after: {inactivity_counter} iteration")
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        observer.stop()
        observer.join()
        print("no new files recorded, script stopped.")

if __name__ == "__main__":
    path = 'C:/Users/s3599678/OneDrive - RMIT University/PhD/misc/Seoul/srcipts'
    time_in = 10  # time check as files come in (seconds)
    no_file_interval = 2  # number of sleeping iterations
    watch_folder(path, time_in, no_file_interval)