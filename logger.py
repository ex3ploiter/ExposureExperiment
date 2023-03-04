import csv
import logging
import os
from datetime import datetime
import json


class Logger:
    def __init__(self, save_dir, exp_name, hparams):
        self.save_dir = save_dir
        self.exp_name = exp_name
        self.hparams = hparams

        # Create directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

        # Initialize metrics logger
        csv_file_name = f"{self.exp_name}_metrics.csv"
        csv_file_path = os.path.join(self.save_dir, csv_file_name)
        csv_file_path = self._get_unique_file_path(csv_file_path)
        self.csv_file = open(csv_file_path, mode="w")
        
        # Initialize txt logger
        log_path_name = f"{self.exp_name}_logs.txt"
        self.log_path = os.path.join(self.save_dir, log_path_name)
        self.log_path = self._get_unique_file_path(self.log_path)
        with open(self.log_path, 'a') as f:
            f.write(f'==== Starting log at {datetime.now()} ====\n')

        self.fig_path = os.path.join(self.save_dir, 'sample_images')
        os.makedirs(self.fig_path, exist_ok=True)

        self.add_hparams(hparams)


    def add_log(self, message):
        with open(self.log_path, 'a') as f:
            f.write(f'{datetime.now()}: {message}\n')
        print(message)

    def add_csv(self, dict_to_append):
        fieldnames = dict_to_append.keys()
        writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        if self.csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(dict_to_append)

    def add_figure(self, fig, name, epoch):
        fig_name = f'{self.exp_name}-epoch-{epoch:02d}-{name}.png'
        filepath = os.path.join(self.fig_path, fig_name)
        fig.savefig(filepath)

    def add_hparams(self, args):
        hparams_path = os.path.join(self.save_dir, 'hparams.json')
        hparams_path = self._get_unique_file_path(hparams_path)

        with open(hparams_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

    def _get_unique_file_path(self, filepath):
        """
        append a unique number to the file name
        to avoid overwriting the existing file.
        """
        base_path, ext = os.path.splitext(filepath)
        i = 0
        while True:
            unique_path = f"{base_path}_{i:02d}{ext}"
            if not os.path.exists(unique_path):
                return unique_path
            i += 1