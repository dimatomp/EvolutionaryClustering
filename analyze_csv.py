import sys
import pandas as pd
from io import StringIO


def read_results(fname):
    with open(fname, 'r') as f:
        content = f.read().split('\n')
    while content[-1] == '':
        content = content[:-1]
    assert content[-3].startswith('Running') or content[-3].startswith('Resulting')
    logs = StringIO('\n'.join(content[:-4]))
    data = pd.read_csv(logs, index_col='generation')
    success = data['index'][1:] != data['index'].data[:-1]
    success = data.iloc[1:][success]
    return data, success


if __name__ == '__main__':
    success = read_results(sys.argv[1])[1]
    pd.set_option('display.max_rows', len(success))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    print(success)
