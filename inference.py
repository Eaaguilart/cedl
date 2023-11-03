import argparse
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from resnet import resnet20, resnet32, resnet44, resnet56
import torch.nn as nn
import timm
from continuum import rehearsal
from utils import MetricLogger, SoftTarget, init_distributed_mode, build_dataset
from losses import edl_log_loss
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from PIL import *
from torchvision import transforms
from glob import iglob
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from torcheval.metrics.functional import binary_auprc
from torch.autograd import Variable

from pytorch_ood.detector import EnergyBased, Mahalanobis, ODIN, MaxLogit, Entropy, MaxSoftmax
from pytorch_ood.utils import OODMetrics

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Class-Incremental Learning training and evaluation script', add_help=False)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_bases', default=10, type=int)
    parser.add_argument('--increment', default=10, type=int)
    parser.add_argument('--backbone', default="resnet32", type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--input_size', default=32, type=int)
    parser.add_argument('--color_jitter', default=0.4, type=float)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--herding_method', default="barycenter", type=str)
    parser.add_argument('--memory_size', default=2000, type=int)
    parser.add_argument('--fixed_memory', default=False, action="store_true")
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--num_epochs', default=140, type=int)
    parser.add_argument('--smooth', default=0.0, type=float)
    parser.add_argument('--eval_every_epoch', default=5, type=float)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--data_set', default='cifar')
    parser.add_argument('--data_path', default='data/cifar100')
    parser.add_argument('--lambda_kd', default=0.5, type=float)
    parser.add_argument('--dynamic_lambda_kd', action="store_true")
    parser.add_argument('--episode', default=0, type=int)
    parser.add_argument('--model_id', default=0, type=int)
    return parser


def init_seed(args):
    return
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_parameters(m, requires_grad=False):
    if m is None:
        return

    if isinstance(m, nn.Parameter):
        m.requires_grad = requires_grad
    else:
        for p in m.parameters():
            p.requires_grad = requires_grad


def get_backbone(args):
    if args.backbone == "resnet32":
        backbone = resnet32()
    elif args.backbone == "resnet20":
        backbone = resnet20()
    elif args.backbone == "resnet44":
        backbone = resnet44()
    elif args.backbone == "resnet56":
        backbone = resnet56()
    else:
        raise NotImplementedError(f'Unknown backbone {args.model}')

    return backbone


class CilClassifier(nn.Module):
    def __init__(self, embed_dim, nb_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = nn.ModuleList([nn.Linear(embed_dim, nb_classes).cuda()])

    def __getitem__(self, index):
        return self.heads[index]

    def __len__(self):
        return len(self.heads)

    def forward(self, x):
        logits = torch.cat([head(x) for head in self.heads], dim=1)
        return logits

    def adaption(self, nb_classes):
        self.heads.append(nn.Linear(self.embed_dim, nb_classes).cuda())


class CilModel(nn.Module):
    def __init__(self, backbone):
        super(CilModel, self).__init__()
        self.backbone = get_backbone(backbone)
        self.fc = None

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        out, x = self.forward_bc(x)
        return out, x

    def forward_bc(self, x):
        w = torch.cat([head.weight.data for head in self.fc], dim=0)
        norms = torch.norm(w, dim=1)
        x = self.backbone(x)
        out = self.fc(x)/norms
        return out, x

    def intermediate_forward(self, x, layer_index):
        x = self.backbone.intermediate_forward(x, layer_index)
        return x

    def feature_list(self, x):
        x, feature_list = self.backbone.feature_list(x)
        return self.fc(x), feature_list

    def features(self, x):
        x, _ = self.backbone.feature_list(x)
        return x

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self, names=["all"]):
        freeze_parameters(self, requires_grad=True)
        self.train()
        for name in names:
            if name == 'fc':
                freeze_parameters(self.fc)
                self.fc.eval()
            elif name == 'backbone':
                freeze_parameters(self.backbone)
                self.backbone.eval()
            elif name == 'all':
                freeze_parameters(self)
                self.eval()
            else:
                raise NotImplementedError(
                    f'Unknown module name to freeze {name}')
        return self

    def prev_model_adaption(self, nb_classes):
        if self.fc is None:
            self.fc = CilClassifier(self.feature_dim, nb_classes).cuda()
        else:
            self.fc.adaption(nb_classes)

    def after_model_adaption(self, nb_classes, args):
        if args.task_id > 0:
            self.weight_align(nb_classes)

    @torch.no_grad()
    def weight_align(self, nb_new_classes):
        w = torch.cat([head.weight.data for head in self.fc], dim=0)
        norms = torch.norm(w, dim=1)
        print(norms, norms[:-nb_new_classes], norms[-nb_new_classes:])
        norm_old = norms[:-nb_new_classes]
        norm_new = norms[-nb_new_classes:]

        gamma = torch.mean(norm_old) / torch.mean(norm_new)
        print(f"old norm / new norm ={gamma}")
        self.fc[-1].weight.data = gamma * w[-nb_new_classes:]


@torch.no_grad()
def eval(model, val_loader, K=None, epoch=None):
    metric_logger = MetricLogger(delimiter="  ")
    model.eval()
    for images, target, task_ids in val_loader:
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        logits, _ = model(images)
        loss = edl_log_loss(logits, target, epoch, K, 10, torch.cuda.current_device()) #criterion(logits, target)
        acc1, acc5 = timm.utils.accuracy(
            logits, target, topk=(1, min(5, logits.shape[1])))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    print(' Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return metric_logger.acc1.global_avg

def inference(args, class_order, args_model, is_id = True):
    args.class_order = class_order
    scenario_train, args.nb_classes = build_dataset(is_train=True, apply_da=False, args=args)
    scenario_val, _ = build_dataset(is_train=False, apply_da=False, args=args)
    args_model.increment_per_task = [args_model.num_bases] + \
        [args_model.increment for _ in range(len(scenario_train) - 1)]
    args_model.known_classes = 0

    torch.distributed.barrier()

    model = CilModel(args_model)
    model = model.cuda()
    model_id = args.episode
    args_model.task_id = 0
    model.prev_model_adaption(args_model.increment_per_task[0])

    if model_id!=0:
        for i in range(model_id):
            args_model.task_id += 1
            args_model.known_classes += args_model.increment_per_task[args_model.task_id]
            model.prev_model_adaption(args_model.increment_per_task[i+1])

    model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args_model.rank])
    model.load_state_dict(torch.load("model_"+str(model_id)))

    try:
        interpolation = torch.transforms.functional.InterpolationMode.BICUBIC
    except:
        interpolation = 3

    model = model.module

    def bal(b_i, b_c):
       indices = torch.argwhere(b_i * b_c < 1e-8)[:,0]
       tmp = 1 - (torch.abs(b_i-b_c)/(b_i+b_c+1e-8))
       if len(indices) > 0:
           tmp[indices] = 0
       return tmp

    def diss(alpha, C):
        e = alpha-1
        S = torch.sum(alpha, dim=1, keepdim=True)
        b = e/S
        tot = 0
        for c in range(C):
            b_c = b[:,c]
            acc_n = 0
            acc_d = 0
            for i in range(C):
                if i != c:
                    b_i = b[:,i]
                    acc_n += b_i * bal(b_i, b_c)
                    acc_d += b_i
            tot+= b_c * (acc_n / acc_d)
        return tot

    unc = []
    unc_diss_l = []
    for task_id, dataset_train in enumerate(scenario_val):
        if task_id > args_model.task_id:
            break
        train_sampler = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                                  sampler=train_sampler, num_workers=10, pin_memory=False)
        T = 2

        for idx, (inputs, targets, task_ids) in enumerate(train_loader):

            logits, _ = model(inputs.cuda())
            evidence = torch.exp(logits)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            unc_i = (args_model.num_bases+model_id*args_model.increment)/S.cpu().detach().numpy()
            unc.append(unc_i[:,0])

            unc_diss = diss(alpha, args_model.num_bases+model_id*args_model.increment).detach().cpu().numpy()
            unc_diss_l.append(unc_diss)

        if not is_id:
            break

    return np.concatenate(unc), np.concatenate(unc_diss_l)


def fpr_at_95_tpr(preds, labels):
    """Return the FPR when TPR is at minimum 95%.
    preds: array, shape = [n_samples]
    Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
    labels: array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.
        """
    fpr, tpr, _ = roc_curve(labels, preds)
    if all(tpr   <   0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x>=0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        'Class-Incremental Learning training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    init_distributed_mode(args)

    init_seed(args)
    

    all_classes = [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 
                        40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 
                        58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 
                        13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 
                        91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 
                        37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 
                        42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 
                        88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 
                        62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 
                        51, 48, 73, 93, 39, 67, 29, 49, 57, 33]

    in_classes = all_classes
    out_classes = all_classes[args.num_bases+args.increment*args.episode:]+all_classes[:args.num_bases+args.increment*args.episode]

    in_out_classes = [in_classes, out_classes]

    args_model = copy.deepcopy(args)
    unc_in, unc_d_in = inference(args, in_out_classes[0], args_model)
    unc_ood, unc_d_ood = inference(args, in_out_classes[1], args_model, is_id=False)

    ones = np.ones(100*(args_model.num_bases+args_model.increment*args.episode), dtype=np.int64)
    zeros = np.zeros(100*args_model.num_bases, dtype=np.int64)
    gt = np.concatenate([ones, zeros])

    print("UNC_d: ", roc_auc_score(gt, np.concatenate([1-unc_d_in[:], 1-unc_d_ood])))
    print("UNC_v: ", roc_auc_score(gt, np.concatenate([1-unc_in[:], 1-unc_ood])))
    print("fpr95 UNC_d:", fpr_at_95_tpr(np.concatenate([1-unc_d_in[:], 1-unc_d_ood]),gt))
    print("fpr95 UNC_v:", fpr_at_95_tpr(np.concatenate([1-unc_in[:], 1-unc_ood]),gt))
    precision, recall, thresholds = precision_recall_curve(gt, np.concatenate([1-unc_in[:], 1-unc_ood]))
    print("auc UNC_v:", auc(recall,precision))
    precision, recall, thresholds = precision_recall_curve(gt, np.concatenate([1-unc_d_in[:], 1-unc_d_ood]))
    print("auc UNC_d:", auc(recall,precision))
