import os
import os.path as osp
import math
import argparse
import yaml
import time
import torch

from data import create_dataloader, prepare_data
from models import define_model
from models.networks import define_generator
from metrics.metric_calculator import MetricCalculator
from metrics.model_summary import register, profile_model
from utils import base_utils, data_utils
import cv2

def test(opt):

    test_loader = create_dataloader(opt, dataset_idx=dataset_idx)

    # infer and store results for each sequence
    for i, data in enumerate(test_loader):

        # fetch data
        lr_data = data['lr'][0]
        seq_idx = data['seq_idx'][0]
        frm_idx = [frm_idx[0] for frm_idx in data['frm_idx']]

        # infer
        hr_seq = model.infer(lr_data)  # thwc|rgb|uint8
        # save results (optional)
        if opt['test']['save_res']:
            res_dir = osp.join(opt['test']['res_dir'], ds_name, model_idx)
            res_seq_dir = osp.join(res_dir, seq_idx)
            data_utils.save_sequence(res_seq_dir, hr_seq, frm_idx, to_bgr=True)


if __name__ == '__main__':
    # ----------------- parse arguments ----------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='directory of the current experiment')
    parser.add_argument('--mode', type=str, required=True,
                        help='which mode to use (train|test|profile)')
    parser.add_argument('--model', type=str, required=True,
                        help='which model to use (FRVSR|TecoGAN)')
    parser.add_argument('--opt', type=str, required=True,
                        help='path to the option yaml file')
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='GPU index, -1 for CPU')
    parser.add_argument('--lr_size', type=str, default='3x256x256',
                        help='size of the input frame')
    parser.add_argument('--test_speed', action='store_true',
                        help='whether to test the actual running speed')
    args = parser.parse_args()


    # ----------------- get options ----------------- #
    print(args.exp_dir)
    with open(osp.join(args.exp_dir, args.opt), 'r') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)


    # ----------------- general configs ----------------- #
    # experiment dir
    opt['exp_dir'] = args.exp_dir

    # random seed
    base_utils.setup_random_seed(opt['manual_seed'])

    # logger
    base_utils.setup_logger('base')
    opt['verbose'] = opt.get('verbose', False)

    # device
    if args.gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            opt['device'] = 'cuda'
        else:
            opt['device'] = 'cpu'
    else:
        opt['device'] = 'cpu'

    # logging
    logger = base_utils.get_logger('base')

    frame_cnt = 0
    fps = 30
    size = (320, 134)
    size_sr = (1280, 536)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowriter = cv2.VideoWriter("video/video_source/test2.avi", fourcc, fps, size_sr)
    path = r"video/video_temp/sr_output/Vid4/EGVSR_iter420000/frame/"

    video_cap = cv2.VideoCapture('video/video_source/test1.avi')
    if video_cap.isOpened():
        rval, frame = video_cap.read()
    else:
        rval = False

    # ----------------- test ----------------- #
    if args.mode == 'test':
        # setup paths
        base_utils.setup_paths(opt, mode='test')

        # run
        opt['is_train'] = False

        if opt['verbose']:
            logger.info('{} Configurations {}'.format('=' * 20, '=' * 20))
            base_utils.print_options(opt, logger)

        # infer and evaluate performance for each model
        for load_path in opt['model']['generator']['load_path_lst']:
            # setup model index
            model_idx = osp.splitext(osp.split(load_path)[-1])[0]
            # log
            logger.info('=' * 40)
            logger.info('Testing model: {}'.format(model_idx))
            logger.info('=' * 40)

            # create model
            opt['model']['generator']['load_path'] = load_path
            model = define_model(opt)

            # for each test dataset
            for dataset_idx in sorted(opt['dataset'].keys()):
                # use dataset with prefix `test`
                if not dataset_idx.startswith('test'):
                    continue

                ds_name = opt['dataset'][dataset_idx]['name']
                logger.info('Testing on {}: {}'.format(dataset_idx, ds_name))
                #============================================================
                while rval:
                    frame_cnt += 1
                    rval, frame = video_cap.read()
                    if rval:
                        cv2.imwrite('video/video_temp/lr_input/frame/{}.png'.format(frame_cnt),frame)  # 存储为图像
                        if (frame_cnt == 15 ):
                            frame_cnt = 0
                            test(opt)
                            img_list = os.listdir(path)
                            img_list.sort()
                            img_list.sort(key=lambda x: int(x[:-4]))  ##文件名按数字排序
                            img_nums = len(img_list)
                            for i in range(img_nums):
                                img = path+img_list[i]
                                frame = cv2.imread(img)
                                # print(frame.shape)  # h, w, c (480, 640, 3)
                                videowriter.write(frame)
                video_cap.release()
                videowriter.release()
                #============================================================
                logger.info('-' * 40)

            # logging
            logger.info('Finish testing')
            logger.info('=' * 40)


    else:
        raise ValueError(
            'Unrecognized mode: {} (train|test|profile)'.format(args.mode))
