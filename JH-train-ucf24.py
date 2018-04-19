""" Adapted from:
    @gurkirt faster_rcnn_pytorch: https://github.com/gurkirt/realtime-action-detection
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as torch_data
from data import v2, UCF24Detection, AnnotationTransform, \
                 detection_collate, CLASSES, BaseTransform
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from tqdm import tqdm
import numpy as np
import time
from utils.evaluation import evaluate_detections
from layers.box_utils import decode, nms
from utils.extentions import AverageMeter, str2bool, save_checkpoint
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
# tensorboard for pytorch
from tensorboardX import SummaryWriter

# parser
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--baseself-net', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--dataset', default='ucf24', help='pretrained base model')
parser.add_argument('--ssd_dim', default=300, type=int, help='Input Size for SSD') # only support 300 now
parser.add_argument('--input_type', default='rgb', type=str, help='INput tyep default rgb options are [rgb,brox,fastOF]')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--nb_epoches', default=65, type=int, help='Number of training epoch')
parser.add_argument('--man_seed', default=123, type=int, help='manualseed for reproduction')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--data_root', default='/mnt/home/jeffrey/data/', help='Location of UCF24 root directory')
parser.add_argument('--save_root', default='/mnt/home/jeffrey/workspace/realtime-action-detection/', help='Location to save checkpoint models')
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.01, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--topk', default=50, type=int, help='topk for evaluation')


class ActionDetection_module():
    def __init__(self):
        # Parse arguments
        self.args = parser.parse_args()
        # set random seeds
        np.random.seed(self.args.man_seed)
        torch.manual_seed(self.args.man_seed)
        if self.args.cuda:
            torch.cuda.manual_seed_all(self.args.man_seed)

        torch.set_default_tensor_type('torch.FloatTensor')

        # set the parameters for this experiment
        self.args.cfg = v2
        self.args.train_sets = 'train'
        self.args.means = (104, 117, 123)
        num_classes = len(CLASSES) + 1
        self.args.num_classes = num_classes
        # self.args.loss_reset_step = 30
        self.args.eval_step = 10000
        self.args.print_step = 1000

        # the name of experiment
        self.args.exp_name = 'CONV-SSD-{}-{}-bs-{}-{}-lr-{:05d}'.format(
            self.args.dataset,
            self.args.input_type,
            self.args.batch_size,
            self.args.baseself_net[:-14],
            int(self.args.lr*100000)
        )
        # save root for tensorboard
        log_dir = os.path.join(
            self.args.save_root+'runs/'+self.args.exp_name+'/'
        )
        # self.args.save_root += self.args.dataset+'/'
        self.args.save_root = self.args.save_root+'cache/'+self.args.exp_name+'/'

        if not os.path.isdir(self.args.save_root):
            os.makedirs(self.args.save_root)
        
        self.tensorboard = SummaryWriter(log_dir=log_dir)

        # Display and Record the experiment info in "training.log"
        self.log_file = open(self.args.save_root+"training.log", "w", 1)
        self.log_file.write(self.args.exp_name+'\n')
        # Display all of the values in self.args
        for arg in vars(self.args):
            print("[{}]: {}".format(arg, getattr(self.args, arg)))
            self.log_file.write(
                str(arg)+': '+str(getattr(self.args, arg))+'\n'
            )

    def build_model(self):

        self.net = build_ssd(300, self.args.num_classes)

        if self.args.cuda:
            self.net = self.net.cuda()

        def xavier(param):
            init.xavier_uniform(param)

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                m.bias.data.zero_()

        print('Initializing weights for extra layers and HEADs...')
        # initialize newly added layers' weights with xavier method
        self.net.extras.apply(weights_init)
        self.net.loc.apply(weights_init)
        self.net.conf.apply(weights_init)

        if self.args.input_type == 'fastOF':
            print('Download pretrained brox flow trained model weights and place them at:::=> ',self.args.data_root + '/train_data/brox_wieghts.pth')
            pretrained_weights = \
                self.args.data_root + '/train_data/brox_wieghts.pth'
            print('Loading base network...')
            self.net.load_state_dict(torch.load(pretrained_weights))
        else:
            vgg_weights = torch.load(
                os.path.join(
                    self.args.data_root,
                    '/ucf24/train_data/',
                    self.args.baseself_net
                )
            )
            print('Loading base network...')
            self.net.vgg.load_state_dict(vgg_weights)

        self.args.data_root += self.args.dataset + '/'
        
    def loss_and_optimizer(self):
        # Get parmeter of self.network in dictionary format with name being key
        parameter_dict = dict(self.net.named_parameters())
        params = []

        # Set different learning rate to bias layers 
        # and set their weight_decay to 0
        for name, param in parameter_dict.items():
            if name.find('bias') > -1:
                # print(name, 'layer parameters will be trained @ {}'.format(self.args.lr*2))
                params += [{'params': [param], 'lr': self.args.lr*2, 'weight_decay': 0}]
            else:
                # print(name, 'layer parameters will be trained @ {}'.format(self.args.lr))
                params += [{'params': [param], 'lr': self.args.lr, 'weight_decay':self.args.weight_decay}]

        self.optimizer = optim.SGD(
            params,
            lr=self.args.lr,
            momentum=self.args.momentum, 
            weight_decay=self.args.weight_decay
        )
        self.criterion = MultiBoxLoss(
            self.args.num_classes,
            overlap_thresh=0.5,
            prior_for_matching=True,
            bkg_label=0,
            neg_mining=True,
            neg_pos=3,
            neg_overlap=0.5,
            encode_target=False,
            use_gpu=self.args.cuda
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'max', patience=1, verbose=True)
 
    def load_dataset(self):
        print('Loading Dataset...')
        train_dataset = UCF24Detection(
            root=self.args.data_root,
            image_set=self.args.train_sets,
            transform=SSDAugmentation(self.args.ssd_dim, self.args.means),
            target_transform=AnnotationTransform(),
            input_type=self.args.input_type
        )
        self.val_dataset = UCF24Detection(
            root=self.args.data_root,
            image_set='test',
            transform=BaseTransform(self.args.ssd_dim, self.args.means),
            target_transform=AnnotationTransform(),
            input_type=self.args.input_type,
            full_test=False
        )

        self.train_loader = torch_data.DataLoader(
            train_dataset,
            self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            collate_fn=detection_collate,
            pin_memory=True
        )
        self.val_loader = torch_data.DataLoader(
            self.val_dataset,
            self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn=detection_collate,
            pin_memory=True
        )

        # epoch_size = len(self.train_dataset) // self.args.batch_size
        # print('Training SSD on', self.train_dataset.name)

    def train_1epoch(self):
        # loss counters
        batch_time = AverageMeter()
        losses = AverageMeter()
        loc_losses = AverageMeter()
        cls_losses = AverageMeter()

        # switch to train model
        self.net.train()

        des = 'Epoch:[{}/{}][train]'.format(self.epoch, self.args.nb_epoches)
        progress = tqdm(self.train_loader, ascii=True, desc=des)
        
        for step, (images, targets, img_indexs) in enumerate(progress):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.iterations += 1

            if self.args.cuda:
                images = Variable(images.cuda())
                targets = [
                    Variable(anno.cuda(), volatile=True) for anno in targets
                ]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            # forward
            out = self.net(images)

            # backprop
            self.optimizer.zero_grad()

            # loss
            loss_l, loss_c = self.criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            self.optimizer.step()

            loc_loss = loss_l.data[0]
            conf_loss = loss_c.data[0]

            # update loss
            loc_losses.update(loc_loss)
            cls_losses.update(conf_loss)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_time.update(t1 - t0)

            losses.update(
                (loc_loss + conf_loss)/2.0
            )

            # tqdm visualization
            info = {
                'iter': '{:06d}'.format(self.iterations),
                'loc-loss': '{:.2f}'.format(loc_losses.avg),
                'cls-loss': '{:.2f}'.format(cls_losses.avg),
                'avg-loss': '{:.2f}'.format(losses.avg),
                # 'time': '{:.2f}'.format(batch_time.avg)
            }
            progress.set_postfix(info)

            if step % self.args.print_step == 0:
                # print on log file
                print_line = \
                    'Epoch[{}][{:06d}/{:06d}], Iterations: {}'.format(
                        self.epoch,
                        step,
                        len(self.train_loader),
                        self.iterations
                    )
                self.log_file.write(print_line)
                for k, v in info.items():
                    self.log_file.write(' [{}]: {}'.format(k, v))
                self.log_file.write('\n')
        # tensorboard utils
        tb_info = {
            'Batch Time': batch_time.avg,
            'loc-loss': '{:.3f}'.format(loc_losses.avg),
            'cls-loss': '{:.3f}'.format(cls_losses.avg),
            'avg-loss': '{:.3f}'.format(losses.avg),
            'lr': self.optimizer.param_groups[0]['lr']
        }
        for k, v in tb_info.items():
            self.tensorboard.add_scalar('train/' + k, v, self.iterations)

    def val_1epoch(self):
        # batch_time = AverageMeter()
        losses = AverageMeter()
        loc_losses = AverageMeter()
        cls_losses = AverageMeter()
        # switch net to evaluation mode
        self.net.eval()
        val_step = 10
        
        det_boxes = [[] for _ in range(len(CLASSES))]
        gt_boxes = []
        count = 0
        torch.cuda.synchronize()

        des = 'Epoch:[{}/{}][val]'.format(self.epoch, self.args.nb_epoches)
        progress = tqdm(self.val_loader, ascii=True, desc=des)

        for step, (images, targets, img_indexs) in enumerate(progress):
            num_classes = self.args.num_classes

            if step % val_step == 0:
                torch.cuda.synchronize()
                t0 = time.perf_counter()

            batch_size = images.size(0)
            height, width = images.size(2), images.size(3)
            if self.args.cuda:
                images = Variable(images.cuda(), volatile=True)
                targets_cuda = [
                    Variable(anno.cuda(), volatile=True) for anno in targets
                ]

            out = self.net(images)
            
            loss_l, loss_c = self.criterion(out, targets_cuda)

            loc_loss = loss_l.data[0]
            conf_loss = loss_c.data[0]

            # update loss
            loc_losses.update(loc_loss)
            cls_losses.update(conf_loss)

            losses.update(
                (loc_loss + conf_loss)/2.0
            )
            
            loc_data, conf_preds, prior_data = out

            if step % val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()

            for b in range(batch_size):
                
                count += 1
                gt = targets[b].numpy()
                gt[:, 0] *= width
                gt[:, 2] *= width
                gt[:, 1] *= height
                gt[:, 3] *= height
                gt_boxes.append(gt)
                decoded_boxes = decode(loc_data[b].data, prior_data.data, self.args.cfg['variance']).clone()
                conf_scores = self.net.softmax(conf_preds[b]).data.clone()

                for cl_ind in range(1, num_classes):
                    scores = conf_scores[:, cl_ind].squeeze()
                    c_mask = scores.gt(self.args.conf_thresh)  # greater than minmum threshold
                    scores = scores[c_mask].squeeze()
                    # print('scores size',scores.size())
                    if scores.dim() == 0:
                        # print(len(''), ' dim ==0 ')
                        det_boxes[cl_ind - 1].append(np.asarray([]))
                        continue
                    boxes = decoded_boxes.clone()
                    l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                    boxes = boxes[l_mask].view(-1, 4)
                    # idx of highest scoring and non-overlapping boxes per class
                    ids, counts = nms(boxes, scores, self.args.nms_thresh, self.args.topk)  # idsn - ids after nms
                    scores = scores[ids[:counts]].cpu().numpy()
                    boxes = boxes[ids[:counts]].cpu().numpy()
                    # print('boxes sahpe',boxes.shape)
                    boxes[:,0] *= width
                    boxes[:,2] *= width
                    boxes[:,1] *= height
                    boxes[:,3] *= height

                    for ik in range(boxes.shape[0]):
                        boxes[ik, 0] = max(0, boxes[ik, 0])
                        boxes[ik, 2] = min(width, boxes[ik, 2])
                        boxes[ik, 1] = max(0, boxes[ik, 1])
                        boxes[ik, 3] = min(height, boxes[ik, 3])

                    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
                    det_boxes[cl_ind-1].append(cls_dets)

            if step % val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                # print('NMS stuff Time {:0.3f}'.format(te - tf))
                
                info = {
                    # 'det_img': '{:05d}/{:05d}'.format(count, len(self.val_dataset)),
                    'gpu time': '{:.2f}'.format(tf-t0),
                    'cpu time': '{:.2f}'.format(te-tf),
                    'loc-loss': '{:.2f}'.format(loc_losses.avg),
                    'cls-loss': '{:.2f}'.format(cls_losses.avg),
                    'avg-loss': '{:.2f}'.format(losses.avg),
                }
                progress.set_postfix(info)
     
        mAP, ap_all, ap_strs = evaluate_detections(
            gt_boxes,
            det_boxes,
            CLASSES,
            iou_thresh=0.5
        )

        tb_info = {
            'loc-loss': '{:.3f}'.format(loc_losses.avg),
            'cls-loss': '{:.3f}'.format(cls_losses.avg),
            'avg-loss': '{:.3f}'.format(losses.avg),
            'mAP': '{:.3f}'.format(mAP),
        }
        for k, v in tb_info.items():
            self.tensorboard.add_scalar('val/' + k, v, self.iterations)

        # write info in log file
        print_line = \
            'Epoch[{}][val], Iterations: {}'.format(self.epoch)
        self.log_file.write(print_line)
        for k, v in tb_info.items():
            self.log_file.write(' [{}]: {}'.format(k, v))
        self.log_file.write('\n')
    
        return mAP, ap_all, ap_strs

    def run(self):
        # call function
        self.build_model()
        self.load_dataset()
        self.loss_and_optimizer()

        # Write model in "training.log"
        self.log_file.write(str(self.net))

        self.best_mAP = 0
        self.iterations = 0
        for self.epoch in range(self.args.nb_epoches):
            # train
            self.train_1epoch()
            # val
            mAP, ap_all, ap_strs = self.val_1epoch()
            self.scheduler.step(mAP)

            for ap_str in ap_strs:
                self.log_file.write(ap_str+'\n')

            # Save the current model and the best model
            is_best = mAP > self.best_mAP
            if is_best:
                save_checkpoint({
                    'epoch': self.epoch,
                    'state_dict': self.net.state_dict(),
                    'best_prec1': self.best_mAP,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, folder=self.args.save_root)


def main():
    action_detection_module = ActionDetection_module()
    action_detection_module.run()


if __name__ == '__main__':
    main()
