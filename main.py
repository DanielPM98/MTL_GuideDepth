import os
import argparse

from training import Trainer
# from train_baseline import Trainer
from evaluate import Evaluator

def get_args():
    file_dir = os.path.dirname(__file__) #Directory of this path

    parser = argparse.ArgumentParser(description='UpSampling for Monocular Depth Estimation')

    #Mode
    parser.set_defaults(train=False)
    parser.set_defaults(evaluate=False)
    parser.add_argument('--train',
                        dest='train',
                        action='store_true')
    parser.add_argument('--eval',
                        dest='evaluate',
                        action='store_true')

    #Data
    parser.add_argument('--data_path',
                        type=str,
                        help='path to train data',
                        default=os.path.join(file_dir, 'dataset'))
    parser.add_argument('--dataset',
                        type=str,
                        help='dataset for training',
                        # choices=['kitti', 'nyu', 'nyu_reduced'],
                        default='nyu')
    parser.add_argument('--resolution',
                        type=str,
                        help='Resolution of the images for training',
                        choices=['full', 'half', 'mini', 'tu_small', 'tu_big'],
                        default='half')
    parser.add_argument('--eval_mode',
                        type=str,
                        help='Eval mode',
                        choices=['alhashim', 'tu'],
                        default='alhashim')


    #Model
    parser.add_argument('--model',
                        type=str,
                        help='name of the model to be trained',
                        default='UpDepth')
    parser.add_argument('--weights_path',
                        type=str,
                        help='path to model weights')

    #Checkpoint
    parser.add_argument('--load_checkpoint',
                        type=str,
                        help='path to checkpoint',
                        default='')
    parser.add_argument('--save_checkpoint',
                        type=str,
                        help='path to save checkpoints to',
                        default='checkpoints')
    parser.add_argument('--save_results',
                        type=str,
                        help='path to save results to',
                        default='./logs')
    parser.add_argument('--name',
                        type=str,
                        help='Name of the testing file',
                        default='default')

    #Optimization
    parser.add_argument('--batch_size',
                        type=int,
                        help='batch size',
                        default=8)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate',
                        default=1e-4)
    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of epochs',
                        default=20)
    parser.add_argument('--scheduler_step_size',
                        type=int,
                        help='step size of the scheduler',
                        default=15)
    parser.add_argument('--strategy',
                        type=str,
                        choices= ['EW', 'RW', 'UW', 'STW'],
                        default='EW',
                        help= 'Weighting strategies')
    parser.add_argument('--mode',
                        type=str,
                        help='Set model mode: single task or dual',
                        choices= ['depth', 'seg', 'dual'],
                        default='dual')

    #System
    parser.add_argument('--num_workers',
                        type=int,
                        help='number of dataloader workers',
                        default=2)


    return parser.parse_args()


def main():
    args = get_args()
    print(args)

    if args.train:
        model_trainer = Trainer(args)
        model_trainer.train()
        args.weights_path = os.path.join(args.save_results, f'{args.name}_results/train/best_model.pth')

    if args.evaluate:
        evaluation_module = Evaluator(args)
        evaluation_module.evaluate()

if __name__ == '__main__':
    main()
