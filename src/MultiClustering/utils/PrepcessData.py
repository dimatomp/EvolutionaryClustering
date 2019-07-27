import pandas as pd
from os import walk
from sklearn import preprocessing as pr
import numpy as np


def sep_by_space():
    for (dirpath, dirnames, files) in walk('datasets/new/sp_by_space'):
        for ii in range(0, len(files)):
            filename = files[ii]
            data = pd.read_csv('datasets/new/sp_by_space/' + filename, delim_whitespace=True)
            data.to_csv('datasets/new/sp_by_space/normal/' + filename, index=False)


def sep_by_sm():
    for (dirpath, dirnames, files) in walk('datasets/new/sp_by_semicol'):
        for ii in range(0, len(files)):
            filename = files[ii]
            data = pd.read_csv('datasets/new/sp_by_semicol/' + filename, delimiter=';')
            data.to_csv('datasets/new/sp_by_semicol/normal/' + filename, index=False)


def fill_missing():
    for (dirpath, dirnames, files) in walk('datasets/new/missing'):
        for ii in range(0, len(files)):
            filename = files[ii]
            data = pd.read_csv('datasets/new/missing/' + filename, header=0)
            rows, columns = data.shape
            for i in range(0, rows):
                for j in range(0, columns):
                    tmp = data.iloc[i, j]
                    if (isinstance(tmp, str)):
                        if ('?' in tmp):
                            data.iloc[i, j] = -1
            data.to_csv('datasets/new/missing/fill/' + filename, index=False)


def del_class0(input_path, output_path):
    for (dirpath, dirnames, files) in walk(input_path):
        for ii in range(0, len(files)):
            filename = files[ii]
            data = pd.read_csv(input_path + '/' + filename, header=None)
            data.drop(data.columns[0], axis=1, inplace=True)
            data.to_csv(output_path + '/' + filename, index=False, header=None)


def del_class1(input_path, output_path):
    for (dirpath, dirnames, files) in walk(input_path):
        print(str(files))
        for ii in range(0, len(files)):
            filename = files[ii]
            data = pd.read_csv(input_path + '/' + filename, header=None)
            rows, columns = data.shape
            data.drop(data.columns[columns - 1], axis=1, inplace=True)
            data.to_csv(output_path + '/' + filename, index=False, header=None)
        break

def del_class(input_path, filename, col, output_path):
    data = pd.read_csv(input_path + "/" + filename, header=None)
    print(data.shape)
    rows, columns = data.shape
    data.drop(data.columns[col], axis=1, inplace=True)
    data.to_csv(output_path + '/' + filename, index=False, header=None)



def scale_features(input_path, output_path):
    for (dirpath, dirnames, files) in walk(input_path):
        print(str(files))
        for ii in range(0, len(files)):
            filename = files[ii]
            data = pd.read_csv(input_path + '/' + filename, sep=',', header=None)
            df_norm = (data - data.mean()) / (data.max() - data.min())
            df_norm.to_csv(output_path + '/' + filename, index=False, header=None)
        break


def encode_features(input_path, filename, output_path, columns: [int]):
    data = pd.read_csv(input_path + '/' + filename, sep=',', header=None)
    for c in columns:
        le = pr.LabelEncoder()
        c_name = data.columns[c]
        values = data[c_name].apply(lambda x: str(x))
        le.fit(values)
        data[c_name] = data[c_name].apply(lambda x: le.transform([x])[0])
    data.to_csv(output_path + '/' + filename, index=False, header=None)

# uncomment necessary calls before run.

# del_class1('datasets/exp', 'datasets/no_class')
# del_class0('datasets/new/class0', 'datasets/no_class')
# del_class('datasets/exp', 'leaf[0].csv', 0 ,'datasets/no_class')
# encode_features('datasets/no_class', 'ecoli.csv', 'datasets/encoded', columns=[0,1,2,3])
# encode_features('datasets/no_class', 'flags.csv', 'datasets/encoded', columns=[16,-1,-2])
# encode_features('datasets/no_class', 'forestfires.csv', 'datasets/encoded', columns=[2,3])
# fill_missing()
# scale_features('datasets/encoded', 'datasets/norm_new')
