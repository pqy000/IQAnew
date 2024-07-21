import torch
from scipy import stats
import numpy as np
import model
import LBP_and_GFNnet
# import models
import data_loader

class LBPIQASolver(object):

    def __init__(self, config, path, train_idx, test_idx):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.path = path  # 将path存储为LBPIQASolver的属性

        self.model_lbp = LBP_and_GFNnet.BaseModel().cuda()
        self.model_lbp.train(True)
        self.run = config.run
        self.l1_loss = torch.nn.SmoothL1Loss().cuda()
        self.l2_loss = torch.nn.MSELoss().cuda()
        self.backbone_params = list(map(id, self.model_lbp.efficientnet.parameters()))
        self.backbone_params += list(map(id, self.model_lbp.efficientnet1.parameters()))
        self.lbp_params = filter(lambda p: id(p) not in self.backbone_params, self.model_lbp.parameters())
        self.backbone = filter(lambda p: id(p) in self.backbone_params, self.model_lbp.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        self.wb = config.wb
        temp = self.model_lbp.parameters()
        paras = [{'params': self.lbp_params, 'lr': self.lr * self.lrratio},
                 {'params': self.backbone, 'lr': self.lr}]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        pred_scores_file = open("pred_scores.txt", "w")
        gt_scores_file = open("gt_scores.txt", "w")
        best_srcc = 0.0
        best_plcc = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []
            paths = []  # 用于存储路径信息

            for img, label in self.train_data:
                img = img.cuda()
                label = label.cuda()

                self.solver.zero_grad()

                pred = self.model_lbp(img)

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                # logit1, logit2 = logit.chunk(2, dim=0)
                # 将路径信息添加到paths列表中
                paths.append(self.path)

                loss1 = self.l1_loss(pred.squeeze(), label.float().detach())
                # loss2 = self.l2_loss(logit1, logit2)

                self.run.log({'loss_l1': loss1})

                loss = loss1
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

            # 将路径、gt_scores和pred_scores写入文件
            for i in range(len(paths)):
                pred_score = pred_scores[i]
                gt_score = gt_scores[i]
                image_path = paths[i]  # 从paths中获取路径信息
                pred_scores_file.write(f"{image_path}\t{gt_score}\t{pred_score}\n")
        
            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            self.paras = [{'params': self.lbp_params, 'lr': lr * self.lrratio},
                          {'params': self.backbone, 'lr': self.lr}]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_lbp.train(False)
        pred_scores = []
        gt_scores = []

        for img, label in data:
            # Data.
            img = img.cuda()
            label = label.cuda()

            logit, x = self.model_lbp(img)

            pred_scores.append(float(x.item()))
            gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        self.run.log({'test_srcc': test_srcc, 'test_plcc': test_plcc})

        self.model_lbp.train(True)
        return test_srcc, test_plcc
