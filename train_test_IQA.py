import os
import argparse
import random
import numpy as np
from LBPSolver import LBPIQASolver
# import torch
import wandb
import copy
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0,1,2])) # 一般在程序开头设置
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
# net = torch.nn.DataParallel(model)
main_path = "."

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except:
        print('Could not set cuda seed.')
        pass

def main(config):
    folder_path = {
        'live': main_path + '/image_data/LIVE/',  #
        'csiq': main_path + '/image_data/CSIQ/',  #
        'tid2013': main_path + '/image_data/tid2013',
        'livec': main_path + '/image_data/ChallengeDB_release',  #
        'koniq': main_path + '/image_data/koniq/',  #
        'bid': main_path + '/BID/BID/',  #
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq': list(range(0, 10073)),
        'bid': list(range(0, 586)),
    }
    sel_num = img_num[config.dataset]
    # sel_num_test = img_num[config.dataset2]

    srcc_all = np.zeros(config.train_test_num, dtype=np.float64)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float64)
    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))

    if config.wb:
        config.run = wandb.init(project=config.dataset+"_iqa", config=config, mode="online")
    else:
        config.run = wandb.init(project=config.dataset + "_iqa", config=config, mode="disabled")

    config.run.name = str(config.seed)
    # seed = copy.deepcopy(config['seed'])
    set_random_seed(config.seed)

    for i in range(config.train_test_num):
        ###
        print('Round %d' % (i+1))
        # Randomly select 80% images for training and the rest for testing
        random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

        solver = LBPIQASolver(config, folder_path[config.dataset], train_index, test_index)
        srcc_all[i], plcc_all[i] = solver.train()

    # print(srcc_all)
    # print(plcc_all)
    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    config.run.log({'srcc_med': srcc_med, 'plcc_med': plcc_med})
    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))


    #     print('Round %d' % (i+1))
    #     shuffled_img_num1 = img_num[config.dataset][:]
    #     shuffled_img_num2 = img_num[config.dataset2][:]
    # # 打乱这些列表
    #     random.shuffle(shuffled_img_num1)
    #     random.shuffle(shuffled_img_num2)
    # # 使用打乱后的列表进行训练和测试索引的选择
    #     train_index = shuffled_img_num1[:int(round(0.8 * len(shuffled_img_num1)))]
    #     test_index = shuffled_img_num1[int(round(0.8 * len(shuffled_img_num1))):]
    #     solver = LBPIQASolver(config, folder_path[config.dataset], folder_path[config.dataset2], train_index, test_index)
    #     srcc_all[i], plcc_all[i] = solver.train()
    # srcc_med = np.median(srcc_all)
    # plcc_med = np.median(plcc_all)
    # print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))
    # srcc_med = np.median(srcc_all)
    # plcc_med = np.median(plcc_all)

    # print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))
    ###交叉验证
        # Randomly select 80% images for training and the rest for testing
        # random.shuffle(sel_num)
        # train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        # test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

        # train_index = random.shuffle(img_num[config.dataset])
        # print(train_index)
        # test_index = random.shuffle(img_num[config.dataset2])

        # solver = LBPIQASolver(config, folder_path[config.dataset],folder_path[config.dataset2],train_index, test_index)
        # srcc_all[i], plcc_all[i] = solver.train()


    # print(srcc_all)
    # print(plcc_all)
   
    # return srcc_med, plcc_med


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='csiq', help='Support datasets: livec|koniq|bid|live|csiq|tid2013')
    # parser.add_argument('--dataset2', dest='dataset2', type=str, default='tid2013', help='Support datasets: livec|koniq|bid|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')
    parser.add_argument('--wb', type=bool, default=True, help='use wandb')
    parser.add_argument('--seed', type=int, default=2, help='use wandb')

    config = parser.parse_args()

    main(config)


