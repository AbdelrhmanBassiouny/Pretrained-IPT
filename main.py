# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

from option import args
import numpy as np
import torch
import utility
import data
import loss
from trainer import Trainer
import warnings
warnings.filterwarnings('ignore')
import os
os.system('pip install einops')
import model
torch.manual_seed(args.seed)
np.random.seed(0)
checkpoint = utility.checkpoint(args)
checkpoint_train = utility.checkpoint(args, name='train')

def get_model():
    if checkpoint.ok:
        _model = model.Model(args, checkpoint)
        if args.pretrain != "":
            if 's3' in args.pretrain:
                import moxing as mox
                mox.file.copy_parallel(args.pretrain, "/cache/models/ipt.pt")
                args.pretrain = "/cache/models/ipt.pt"
            state_dict = torch.load(args.pretrain)
            _model.model.load_state_dict(state_dict, strict= False)
        if len(args.freeze) >= 1:
            to_freeze = args.freeze.split('+')
            freeze_dict = {'h':'head', 'b':'body', 't':'tail'}
            freeze_list = [freeze_dict[l] for l in to_freeze]
            for param in _model.named_parameters():
                print(param[0])
                # if param[0].split('.')[1] == "body":
                #   param[1].requires_grad = False

        return _model

def main():
    global model
    if checkpoint.ok and checkpoint_train.ok:
        loader = data.Data(args)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        _model = get_model()
        t = Trainer(args, loader, _model, _loss, checkpoint, ckp_train=checkpoint_train)
        if not args.test_only:
            while not t.terminate():
                t.train()
                t.test()
        else:
            t.test()
        checkpoint.done()
            
if __name__ == '__main__':
    main()
