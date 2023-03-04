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
        self.save_dir = self._get_unique_path(os.path.join(self.save_dir, 'run'))

        # Create directory if it does not exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize metrics logger
        csv_file_name = f"{self.exp_name}_metrics.csv"
        csv_file_path = os.path.join(self.save_dir, csv_file_name)
        self.csv_file = open(csv_file_path, mode="w")
        
        # Initialize txt logger
        log_path_name = f"{self.exp_name}_logs.txt"
        self.log_path = os.path.join(self.save_dir, log_path_name)
        with open(self.log_path, 'a') as f:
            f.write(f'==== Starting log at {datetime.now()} ====\n')

        self.fig_path = os.path.join(self.save_dir, 'sample_perturbations')
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
        self.csv_file.flush()

    def add_figure(self, fig, name, epoch):
        fig_name = f'{self.exp_name}-epoch-{epoch:02d}-{name}.png'
        filepath = os.path.join(self.fig_path, fig_name)
        fig.savefig(filepath)

    def add_hparams(self, args):
        hparams_path = os.path.join(self.save_dir, 'hparams.json')

        with open(hparams_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

    def _get_unique_path(self, base_path):
        """
        append a unique number to the file name
        to avoid overwriting the existing file.
        """
        i = 0
        while True:
            unique_path = f"{base_path}_{i:02d}"
            if not os.path.exists(unique_path):
                return unique_path
            i += 1