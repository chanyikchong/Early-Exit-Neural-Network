import json
import os
import re
import shutil
from typing import Dict

import numpy as np
import pandas as pd

MODE_MAPPING = {
    0: "execution_time",
    1: "memory",
    2: "layers_execution_time",
    3: "layers_output_memory",
    4: "backbone_execution_time",
    5: "backbone_output_memory",
    6: "gate_execution_time",
    7: "gate_output_memory",
}


class ProfilingHelper:
    def __init__(self, save_folder, replace=False):
        self.save_folder = save_folder
        if os.path.exists(self.save_folder) and replace:
            shutil.rmtree(self.save_folder)
        os.makedirs(self.save_folder, exist_ok=True)

    def execution_time(self, result):
        file_path = os.path.join(self.save_folder, "execution_time.txt")
        with open(file_path, 'a', encoding='utf8') as f:
            f.write(f"{result}\n")

    def memory(self, result):
        file_path = os.path.join(self.save_folder, "memory.txt")
        with open(file_path, 'a', encoding='utf8') as f:
            f.write(f"{result['model']},{result['output']}\n")

    def layers_profile(self, mode, result):
        file_path = os.path.join(self.save_folder, f"{mode}.txt")
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf8') as f:
                f.write(f"{','.join([f'{k}' for k in result.keys()])}\n")
        with open(file_path, 'a', encoding='utf8') as f:
            f.write(f"{','.join([f'{k}' for k in result.values()])}\n")

    def summarise(self, save=False):
        files = os.listdir(self.save_folder)
        summary = dict()
        for mode in MODE_MAPPING.values():
            if f"{mode}.txt" in files:
                if mode == 'execution_time':
                    summary[mode] = self._read_model_execution_time()
                elif mode == 'memory':
                    summary[mode] = self._read_model_memory()
                elif 'execution_time' in mode:
                    summary[mode] = self._read_layer_execution_time(mode)
                elif 'memory' in mode:
                    summary[mode] = self._read_layer_memory(mode)
        if save:
            with open(os.path.join(self.save_folder, f"summary.json"), 'w', encoding='utf8') as f:
                json.dump(summary, f, indent=4)
        return summary

    def _read_model_execution_time(self) -> Dict[str, float]:
        with open(os.path.join(self.save_folder, f"execution_time.txt"), 'r', encoding='utf8') as f:
            lines = [float(line.strip()) for line in f.readlines()]
        mean = np.mean(lines)
        std = np.std(lines)
        var = np.var(lines)
        lines = np.where(np.array(lines) > 0, lines, 1e-20)
        log_mean = np.mean(np.log(lines))
        log_std = np.std(np.log(lines))
        log_var = np.var(np.log(lines))
        return {'mean': mean,
                'std': std,
                'var': var,
                'min': min(lines),
                'max': max(lines),
                'log_mean': log_mean,
                'log_std': log_std,
                'log_var': log_var}

    def _read_layer_execution_time(self, mode) -> Dict[str, Dict[str, float]]:
        df = pd.read_csv(os.path.join(self.save_folder, f"{mode}.txt"))
        summary_dict = dict()
        for col in df.columns:
            c = df[col].astype(float).values
            c = np.where(c > 0, c, 1e-20)
            mean = np.mean(c)
            std = np.std(c)
            var = np.var(c)
            log_mean = np.mean(np.log(c))
            log_std = np.std(np.log(c))
            log_var = np.var(np.log(c))
            summary_dict[col] = {'mean': mean,
                                 'std': std,
                                 'var': var,
                                 'min': min(c),
                                 'max': max(c),
                                 'log_mean': log_mean,
                                 'log_std': log_std,
                                 'log_var': log_var}
        return summary_dict

    def _read_model_memory(self) -> Dict[str, float]:
        with open(os.path.join(self.save_folder, f"memory.txt"), 'r', encoding='utf8') as f:
            line = f.readline().strip()
        sizes = line.split(',')
        if 'B' in line:
            self._memory_unit_normalize(sizes)
        return {'model': float(sizes[0]), 'output': float(sizes[1])}

    def _read_layer_memory(self, mode) -> Dict[str, Dict[str, float]]:
        with open(os.path.join(self.save_folder, f"{mode}.txt"), 'r') as file:
            lines = file.readlines()

        keys = lines[0].strip().split(',')
        values = lines[1].strip().split(',')
        if 'B' in values[0]:
            self._memory_unit_normalize(values)
        return {k: float(v) for k, v in zip(keys, values)}

    @staticmethod
    def _memory_unit_normalize(sizes):
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB']
        for i, s in enumerate(sizes):
            # Extract the numeric value and the unit
            value = float(re.search(r"[\d.]+", s).group())
            unit = re.search(r"[a-zA-Z]+", s).group()

            unit_idx = units.index(unit)
            sizes[i] = value * (1024 ** unit_idx)
