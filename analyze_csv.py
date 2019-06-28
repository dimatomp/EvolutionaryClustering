import sys
import pandas as pd
import os
import traceback
from .batch_tasks import *
from io import StringIO
from itertools import chain


def read_results(fname, accept_old_format=False):
    if accept_old_format:
        with open(fname, 'r') as f:
            content = f.read()
        if content.find('Traceback') != -1:
            raise ValueError('File has traceback')
        content = content.split('\n')
        while content[-1] == '':
            content = content[:-1]
        old_format = content[-3].startswith('Running') or content[-3].startswith('Resulting')
        if old_format:
            running_time = None
            for s in content[-3:]:
                if s.startswith('Running time'):
                    running_time = float(s[13:s.find(' seconds')])
            content = content[:-3]
        logs = StringIO('\n'.join(content))
    else:
        logs = open(fname, 'r')
        old_format = False
    with logs:
        data = pd.read_csv(logs, index_col='generation')
        if not old_format:
            running_time = data.iloc[-1]['time']
        return data, running_time


def load_strategy_data(getter, prefix='.'):
    columns = ['dataset'] + list(chain(*([index + '_trivial', index + '_dynamic'] for index, _ in indices)))
    rows = []
    for dataset, _ in datas:
        row = [dataset]
        for index, _ in indices:
            try:
                trivials = read_results(prefix + '/' + get_file_name(index, dataset, 'all_mutations_trivial'))
                row.append(getter(trivials))
            except AssertionError:
                row.append(float('nan'))
            try:
                trivials = read_results(prefix + '/' + get_file_name(index, dataset, 'all_mutations_dynamic'))
                row.append(getter(trivials))
            except AssertionError:
                row.append(float('nan'))
        rows.append(row)
    print(columns)
    print(rows[0])
    return pd.DataFrame(rows, columns=columns)


def load_strategy_indices(prefix='.'):
    return load_strategy_data(lambda trivials: trivials[0].iloc[-1]['index'], prefix=prefix)


def load_strategy_times(prefix='.'):
    return load_strategy_data(lambda x: x[1], prefix=prefix)


def load_all_files(folder):
    datas = []
    for s in os.listdir(folder):
        print(s)
        index, dataset, algo = s.split('-')
        algo = algo[:-4]
        try:
            data, time = read_results(folder + '/' + s)
        except:
            traceback.print_exc()
            continue
        datas.append((index, dataset, algo, data, time))
    return datas



#if __name__ == '__main__':
#    success = read_results(sys.argv[1])[0]
#    pd.set_option('display.max_rows', len(success))
#    pd.set_option('display.max_columns', None)
#    pd.set_option('display.width', 2000)
#    pd.set_option('display.float_format', '{:20,.2f}'.format)
#    pd.set_option('display.max_colwidth', -1)
#    print(success)
