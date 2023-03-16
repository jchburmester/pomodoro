import datetime
import subprocess
import csv
import glob
import os
import tensorflow as tf
import time

class SMICallback(tf.keras.callbacks.Callback):
    '''Callback to log GPU metrics using NVIDIA SMI'''
    
    def __init__(self, interval=1):
        super(SMICallback, self).__init__()
        self.interval = interval
        self.gpu_available = False

        # Check if there is a GPU available (we could also check this with tf.test.is_gpu_available)
        try:
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
            self.gpu_available = bool(output.strip())
            if self.gpu_available:
                print('Using: ',output.strip(),'\n')

        except:
            pass

        if not self.gpu_available:
            print('\n' + 'No GPU available, SMICallback will not log GPU metrics' + '\n')

    def on_train_begin(self, logs=None):
        # Find the latest run folder
        run_folders = glob.glob('runs/*/')
        latest_folder = max(run_folders, key=os.path.getmtime)

        self.log_path = os.path.join(latest_folder, 'logs.csv')

        # Write the header to the CSV file
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'time', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'gpu_power_W', 'gpu_temp_C', 'gpu_util_%'])
        # Log metrics at the beginning of training
        current_time = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logs['loss'], logs['accuracy'], logs['val_loss'], logs['val_accuracy'] = 0, 0, 0, 0
        self._log_metrics('pre', logs, current_time)

    def on_train_end(self, logs=None):
        # Log metrics at the end of training
        time.sleep(5)
        current_time = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logs['loss'], logs['accuracy'], logs['val_loss'], logs['val_accuracy'] = 0, 0, 0, 0
        self._log_metrics('post', logs, current_time)

    def on_epoch_end(self, epoch, logs=None):
        # Get current time
        current_time = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # Write the metrics to the CSV file
        self._log_metrics(epoch, logs, current_time)

    def _log_metrics(self, epoch, logs, current_time):
        # Get GPU metrics using NVIDIA SMI
        if self.gpu_available:
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw,temperature.gpu,utilization.gpu', '--format=csv,noheader'])
            gpu_metrics = output.decode().strip().split(',')
            gpu_power, gpu_temp, gpu_util = [str(m).replace(" ", "").replace("W", "").replace("%", "") for m in gpu_metrics]
        else:
            gpu_power, gpu_temp, gpu_util = 'N/A', 'N/A', 'N/A'

        # Write the metrics to the CSV file
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, current_time, round(logs['loss'],6), round(logs['accuracy'],6), round(logs['val_loss'],6), round(logs['val_accuracy'],6), gpu_power, gpu_temp, gpu_util])

