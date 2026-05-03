import os
import sys

import mne
import numpy as np
import pandas as pd


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ds_link = os.path.join(project_root, 'data', 'raw', 'ds004514')

    print('PROJECT ROOT:', project_root)
    print('DATASET LINK PATH:', ds_link)
    exists = os.path.exists(ds_link)
    print('DATASET FOUND:', exists)
    print('PYTHON EXECUTABLE:', sys.executable)

    if exists:
        print('SETUP SUCCESS')
        return 0
    else:
        print('SETUP FAIL: dataset path not found')
        return 1


if __name__ == '__main__':
    sys.exit(main())
