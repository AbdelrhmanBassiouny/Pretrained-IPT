# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import os

from sympy import get_indices
from data import srdata
import numpy as np

train_indices = None
np.random.seed(0)
class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        global train_indices
        data_range = [r.split('-') for r in args.data_range.split('/')]
        begin, end = int(data_range[0][0]), int(data_range[-1][-1])
        size = end - begin + 1
        if train:
            data_range = data_range[0]
            rng = np.random.default_rng()
            train_indices = rng.permutation(size)
            self.indices = train_indices
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
            if train_indices is not None:
                self.indices = [i for i in range(size) if i not in train_indices]
            else:
                test_begin,test_end = int(data_range[0][0]), int(data_range[-1][-1])
                self.indices = list(range(test_begin, test_end, 1))
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
    
    def get_indices(self, data_range):
        begin, end = list(map(lambda x: int(x), data_range))
        size = end - begin + 1
        rng = np.random.default_rng()
        return rng.permutation(size)

    def _scan(self):            
        # SCALE
        # names_hr, names_lr = super(DIV2K, self)._scan_modified()
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        # SCALE
        # names_lr = names_lr[self.begin - 1:self.end]
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        #if self.input_large: self.dir_lr += 'L'

