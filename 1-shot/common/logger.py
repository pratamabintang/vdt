r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training, modelname = None):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
        if logpath == '': logpath = logtime
        if modelname is None:
            modelname = "HSNet"

        cls.logpath = os.path.join('logs', logpath +"_"+ modelname + '.log')
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath, exist_ok=True)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        
        logging.info("\n:=========== Few-shot Seg. with "+modelname+" ===========")
        
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)
        
    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))
    
    @classmethod
    def save_ckp(cls, optimizer,epoch):
        checkpoint = {
        'optimizer':optimizer.state_dict(),
        "epoch": epoch
        }
        torch.save(checkpoint, os.path.join(cls.logpath, 'best_ckp.pt'))
        cls.info('Checkpoint saved @%d' % (epoch))

    @classmethod
    def log_params(cls, model):
        r"""Report frozen-backbone / learnable / total parameter counts.

        Counts by ``requires_grad`` (not by ``state_dict``), so a frozen SAM2
        trunk is reported as backbone instead of "learnable".
        """
        backbone_param = 0
        learner_param = 0
        total_param = 0
        for name, param in model.named_parameters():
            n = param.numel()
            total_param += n
            if param.requires_grad:
                learner_param += n
            elif name.startswith('backbone'):
                backbone_param += n
        Logger.info('Backbone # param. (frozen): %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % total_param)
