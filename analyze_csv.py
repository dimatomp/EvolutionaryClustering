import sys
import pandas as pd
from batch_tasks import *
from io import StringIO
from itertools import chain


def read_results(fname):
    with open(fname, 'r') as f:
        content = f.read().split('\n')
    while content[-1] == '':
        content = content[:-1]
    assert content[-3].startswith('Running') or content[-3].startswith('Resulting')
    running_time = None
    for s in content[-3:]:
        if s.startswith('Running time'):
            running_time = float(s[13:s.find(' seconds')])
    logs = StringIO('\n'.join(content[:-3]))
    data = pd.read_csv(logs, index_col='generation')
    success = data['index'][1:] != data['index'].data[:-1]
    success = data.iloc[1:][success]
    return data, success, running_time


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
    return load_strategy_data(lambda trivials: trivials[1].iloc[-1]['index'], prefix=prefix)


def load_strategy_times(prefix='.'):
    return load_strategy_data(lambda x: x[2], prefix=prefix)


if __name__ == '__main__':
    success = read_results(sys.argv[1])[1]
    pd.set_option('display.max_rows', len(success))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    print(success)
