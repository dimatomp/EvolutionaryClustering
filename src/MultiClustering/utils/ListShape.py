import numpy as np
import pandas as pd
from .. import Constants
import sys
from os import walk

for (dirpath, dirnames, files) in walk(Constants.experiment_path):
    for ii in range(0, len(files)):
        file = files[ii]
        data = pd.read_csv(Constants.experiment_path + file)
        X = np.array(data, dtype=np.double)
        print("#" + file + ": " + str(X.shape))