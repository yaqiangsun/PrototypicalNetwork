import argparse

import torch
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--load', default='./save/proto-5/max-acc.pth')
    parser.add_argument('--batch', type=int, default=2000)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=30)
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)

    dataset = MiniImageNet('test')
    sampler = CategoriesSampler(dataset.label,
                                args.batch, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8, pin_memory=True)

    model = Convnet().cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    print("model.learnable_scale:",model.learnable_scale)


    ave_acc = Averager()
    
    for i, batch in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        data_shot, data_query = data[:k], data[k:]

        x = model(data_shot)
        x = x.reshape(args.shot, args.way, -1).mean(dim=0)
        p = x

        logits = euclidean_metric(model(data_query), p)

        # modify, to get better result!!!!
        if(args.shot==1):
            feature_dims = 1600 # 1600 for miniimagenet, 64 for omniglot
            # learnable_scale = torch.nn.Parameter(torch.FloatTensor(1).fill_(1.0),requires_grad=True).cuda()
            logits = model.learnable_scale * logits / feature_dims
        else:
            # logits = logits / feature_dims
            logits = model.learnable_scale * logits


        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        if i%100==0:
            print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        
        x = None; p = None; logits = None

