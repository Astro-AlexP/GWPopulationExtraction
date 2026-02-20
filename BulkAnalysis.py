from gwosc.datasets import find_datasets
import bilby
from gwpy.timeseries import TimeSeries
from gwosc import datasets
import numpy as np
import os.path
import os
from os import path
import glob



files = [file for file in glob.glob("../Data/*/*.json")]
print(files)
results = []
for file_name in files:
    result = bilby.result.read_in_result(filename=file_name, result_class=bilby.gw.result.CBCResult)
    results.append(result)

print(len(results))