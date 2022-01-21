# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import utility
import torch
from tqdm import tqdm
from copy import deepcopy

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.ckp_train = deepcopy(ckp)
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log(f'\nEvaluation:(epoch {epoch})')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            i = 0
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                if self.args.derain:
                    for norain, rain, filename in tqdm(d, ncols=80):
                        norain,rain = self.prepare(norain, rain)
                        sr = self.model(rain, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)
                        
                        save_list = [sr, rain, norain]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, norain, scale, self.args.rgb_range
                        ) 
                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, 1)
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    
                    is_best = True if epoch == best else False
                    self.ckp.save(self, epoch, is_best=is_best)
                    
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                    isderain = 0
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)
                
    def train(self):
        torch.set_grad_enabled(True)
        epoch = self.optimizer.get_last_epoch()
        self.ckp_train.write_log(f'\Training:(epoch {epoch})')
        self.ckp_train.add_log(
            torch.zeros(1, len(self.loader_train), len(self.scale))
        )
        self.model.train()
        timer_test = utility.timer()
        if self.args.save_results:
            self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_train):
            i = 0
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                if self.args.derain:
                    for norain, rain, filename in tqdm(d, ncols=80):
                        
                        self.optimizer.zero_grad()
                        norain,rain = self.prepare(norain, rain)
                        assert norain is not None
                        sr = self.model(rain, idx_scale, opt=self.optimizer, loss=self.loss, output=norain)
                        
                    self.ckp_train.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp_train.log.max(0)
                    self.ckp_train.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            # d.dataset.name,
                            "DIV2K_train",
                            scale,
                            self.ckp_train.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                    isderain = 0

        self.ckp_train.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp_train.write_log('Saving...')

        if self.args.save_results:
            self.ckp_train.end_background()

        self.ckp_train.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        self.optimizer.scheduler.step()
        torch.set_grad_enabled(False)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
