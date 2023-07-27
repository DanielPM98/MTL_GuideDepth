import os
import time
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model import loader
from data import datasets
from weighting import strategies
from metrics import AverageMeter, Result
from losses import Depth_Loss, Seg_Loss

max_depths = {
    'kitti': 80.0,
    'nyu' : 10.0,
}

class Trainer():
    def __init__(self, args):
        self.debug = True

        self.create_logs(args)

        self.epoch = 0
        self.val_losses = []
        self.max_epochs = args.num_epochs
        self.maxDepth = max_depths[args.dataset]
        print('[INFO] Maximum Depth of Dataset: {}'.format(self.maxDepth))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #TODO: ADAPT WHEN MODEL IS CREATED
        self.model = loader.load_model(args.model,
                                  args.weights_path)
        self.model.to(self.device)

        self.train_loader = datasets.get_dataloader(args.dataset,
                                                 path=args.data_path,
                                                 split='train',
                                                 augmentation=args.eval_mode,
                                                 batch_size=args.batch_size,
                                                 resolution=args.resolution,
                                                 workers=args.num_workers)
        self.val_loader = datasets.get_dataloader(args.dataset,
                                                path=args.data_path,
                                                split='val',
                                                augmentation=args.eval_mode,
                                                batch_size=args.batch_size,
                                                resolution=args.resolution,
                                                workers=args.num_workers)

        self.optimizer = optim.Adam(self.model.parameters(),
                               args.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                 args.scheduler_step_size,
                                                 gamma=0.1)

        if args.eval_mode == 'alhashim':
            self.depth_loss_fn = Depth_Loss(0.1, 1, 1, maxDepth=self.maxDepth)
        else:
            self.depth_loss_fn = Depth_Loss(1, 0, 0, maxDepth=self.maxDepth)

        self.seg_loss_fn = Seg_Loss(self.train_loader.dataset.num_classes, device=self.device)

        self.weighting_strategy = strategies.choose_strategy(args.strategy)

        #Load Checkpoint
        if args.load_checkpoint != '':
            self.load_checkpoint(args.load_checkpoint)

        self.metric_history = []

    def train(self):
        torch.cuda.empty_cache()
        self.start_time = time.time()
        for self.epoch in range(self.epoch, self.max_epochs):
            current_time = time.strftime('%H:%M', time.localtime())
            print('{} - Epoch {}'.format(current_time, self.epoch))

            self.train_loop()

            if self.val_loader is not None:
                self.val_loop()

            self.save_checkpoint()

        self.save_model()
        self.save_results()

    def train_loop(self):
        self.model.train()
        accumulated_loss = 0.0

        for i, data in enumerate(self.train_loader):
            image, gt, label = self.unpack_and_move(data)
            self.optimizer.zero_grad()

            depth_prediction, seg_prediction = self.model(image)

            # Normalize prediction probabilities in range [0, 1]
            seg_prediction = F.softmax(seg_prediction, dim=1)

            # Compute each task loss individually
            depth_loss_value = self.depth_loss_fn(depth_prediction, gt)
            seg_loss_value = self.seg_loss_fn(seg_prediction, label)

            # Compute weighted loss
            loss_value = self.weighting_strategy([depth_loss_value, seg_loss_value])

            loss_value.backward() 

            self.optimizer.step()

            accumulated_loss += loss_value.item()

        #Report 
        current_time = time.strftime('%H:%M', time.localtime())
        average_loss = accumulated_loss / (len(self.train_loader.dataset))
        print('{} - Average Training Loss: {:3.4f}'.format(current_time, average_loss))


    def val_loop(self):
        torch.cuda.empty_cache()
        self.model.eval()
        accumulated_loss = 0.0
        average_meter = AverageMeter()

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                t0 = time.time()
                image, gt, label = self.unpack_and_move(data)
            
                data_time = time.time() - t0

                t0 = time.time()
                inv_depth_prediction, seg_prediction = self.model(image)
                depth_prediction = self.inverse_depth_norm(inv_depth_prediction)

                # Normalize prediction probabilities in range [0, 1]
                seg_prediction = F.softmax(seg_prediction, dim=1)

                gpu_time = time.time() - t0

                if self.debug and i==0:
                    os.makedirs(os.path.join(self.results_pth, 'debug'), exist_ok=True)
                    self.show_images_depth(image, gt, depth_prediction)
                    self.show_images_seg(image, label, seg_prediction)
                    plt.close()

                # Compute each task loss individually
                depth_loss_value = self.depth_loss_fn(inv_depth_prediction, self.depth_norm(gt))
                seg_loss_value = self.seg_loss_fn(seg_prediction, label)

                # Compute weighted loss
                loss_value = self.weighting_strategy([depth_loss_value, seg_loss_value])

                accumulated_loss += loss_value.item()

                result = Result()
                result.evaluate_depth(depth_prediction.data, gt.data)
                result.evaluate_segmentation(seg_prediction.data, label.data)
                average_meter.update(result, gpu_time, data_time, image.size(0))

        #Report 
        avg = average_meter.average()
        current_time = time.strftime('%H:%M', time.localtime())
        average_loss = accumulated_loss / (len(self.val_loader.dataset))

        # Metrics and loss history
        self.val_losses.append(average_loss)
        self.metric_history.append(avg)

        print('{} - Average Validation Loss: {:3.4f}'.format(current_time, average_loss))

        print('\n*\n'
              'RMSE = {average.rmse:.3f}\n'
              'MAE = {average.mae:.3f}\n'
              'Delta1 = {average.delta1:.3f}\n'
              'Delta2 = {average.delta2:.3f}\n'
              'Delta3 = {average.delta3:.3f}\n'
              'REL = {average.absrel:.3f}\n'
              'Lg10 = {average.lg10:.3f}\n'
              '\nMean IoU = {average.mIoU:.3f}\n'
              'MAE = {average.mMAE:.3f}\n'
              'Pixel Accuracy = {average.px_acc}\n\n'
              't_GPU = {time:.3f}\n'.format(
              average=avg, time=avg.gpu_time))


    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path,
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.epoch = checkpoint['epoch']


    def save_checkpoint(self):
        #Save checkpoint for training
        checkpoint_dir = os.path.join(self.checkpoint_pth,
                                      'checkpoint_{}.pth'.format(self.epoch))
        torch.save({
            'epoch': self.epoch + 1,
            'val_losses': self.val_losses,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            }, checkpoint_dir)
        current_time = time.strftime('%H:%M', time.localtime())
        print('{} - Model saved'.format(current_time))


    def save_model(self):
        best_checkpoint_pth = os.path.join(self.checkpoint_pth,
                                      'checkpoint_19.pth')
        best_model_pth = os.path.join(self.results_pth,
                                     'best_model.pth')

        checkpoint = torch.load(best_checkpoint_pth)
        torch.save(checkpoint['model'], best_model_pth)
        print('Model saved.')


    def inverse_depth_norm(self, depth):
        zero_mask = depth == 0.0
        depth = self.maxDepth / depth
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        depth[zero_mask] = 0.0
        return depth


    def depth_norm(self, depth):
        zero_mask = depth == 0.0
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        depth = self.maxDepth / depth
        depth[zero_mask] = 0.0
        return depth


    def unpack_and_move(self, data):
        if isinstance(data, (tuple, list)):
            image = data[0].to(self.device, non_blocking=True)
            gt = data[1].to(self.device, non_blocking=True)
            label = data[2].to(self.device, non_blocking=True)

            if label.dim() == 4: 
                label = torch.squeeze(label, dim=1)

            return image, gt, label
        
        if isinstance(data, dict):
            image = data['image'].to(self.device, non_blocking=True)
            gt = data['depth'].to(self.device, non_blocking=True)
            label = data['label'].to(self.device, non_blocking=True)

            if label.dim() == 4: 
                label = torch.squeeze(label, dim=1)

            return image, gt, label
        
        print('Type not supported')
        exit(0)

    def show_images_depth(self, image, gt, pred):
        image_np = image[0].cpu().permute(1, 2, 0).numpy()
        gt[0, 0, gt[0,0] == 100.0] = 0.1

        save_path = os.path.join(self.results_pth, 'debug/depth_debug.png')
        
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image_np)
        ax[0].set_title('Original Image')

        ax[1].imshow(gt[0, 0].detach().cpu())
        ax[1].set_title('Ground Truth')

        ax[2].imshow(pred[0, 0].detach().cpu())
        ax[2].set_title('Depth Prediction')

        fig.savefig(save_path)


    def show_images_seg(self, image, target, pred):
        image_np = image[0].cpu().permute(1,2,0).numpy()
        pred_np = torch.argmax(pred, dim=1)[0].cpu().numpy()
        target_np = target[0].cpu().numpy()

        # Get number of classes
        num_classes = pred.size(1)

        # Create color map for classes
        cmap = plt.get_cmap('tab20', num_classes)

        # Plot target segmentation
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image_np)
        ax[0].set_title('Original Image')

        ax[1].imshow(target_np, cmap=cmap, vmin=0, vmax=num_classes)
        ax[1].set_title('Ground Truth')

        ax[2].imshow(pred_np, cmap=cmap, vmin=0, vmax=num_classes)
        ax[2].set_title('Segmentation Prediction')

        fig.savefig(os.path.join(self.results_pth, 'debug/seg_debug.png'))
        

    def create_logs(self, args):
        name = os.path.join(args.save_results, f'{args.model}_results')
        self.checkpoint_pth = os.path.join(name, args.save_checkpoint)
        self.results_pth = os.path.join(name, 'train')

        # Create directory for training results output
        os.makedirs(self.results_pth, exist_ok=True)

        # Create directory for saving training checkpoints
        os.makedirs(self.checkpoint_pth, exist_ok=True)

    def save_results(self):
        # Save plot of loss evolution during training
        plt.plot(self.val_losses)
        plt.savefig(os.path.join(self.results_pth, 'training_losses.png'))

        with open(os.path.join(self.results_pth, 'metrics.pkl'), 'wb') as f:
            pickle.dump(self.metric_history, f)


def debug(test):
    print(test)
    print('===== Test Passed =====')
    exit(0)