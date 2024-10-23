import argparse
import logging
import os
import glob
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config
from scipy.ndimage import zoom
import cv2
import copy
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./data/Synapse', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str,
                    default='./output', help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', default=True, action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)


def overlay_segmentation_on_image(original_img, segmentation_mask, alpha=0.5):
    # 調整 segmentation_mask 的大小，使其與原始影像相同
    segmentation_mask_resized = cv2.resize(
        segmentation_mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    # 建立與原始影像大小相同的彩色遮罩
    colored_mask = np.zeros_like(original_img)

    # 使用你原始程式碼的顏色定義
    colors = {
        1: (0, 255, 0),  # 綠色
        2: (255, 0, 0),  # 藍色
        3: (0, 0, 255),  # 紅色
        4: (255, 255, 0) # 黃色
    }

    # 將類別標籤轉換為對應的 RGB 顏色
    for class_label, color in colors.items():
        colored_mask[segmentation_mask_resized == class_label] = color

    # 使用 addWeighted 疊加影像
    overlay = cv2.addWeighted(original_img, 1 - alpha, colored_mask, alpha, 0)
    return overlay





if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 5,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()


    input_W = 224
    input_H = 224
    model_path = '/home/yuchu/Downloads/ronshing_original/Ronshing-001/Ronshing/Swin-Unet/output/epoch_99.pth'
    net.load_state_dict(torch.load(model_path))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patch_size = [input_W,input_H]

    img_dir = '/home/yuchu/Downloads/ronshing_original/Ronshing-001/Ronshing/Swin-Unet/output/predictions/350test.png'
    img = cv2.imread(img_dir)

    x, y, _ = img.shape
    patch_size = [224, 224]

    # 調整影像大小
    if x != patch_size[0] or y != patch_size[1]:
        img = zoom(img, (patch_size[0] / x, patch_size[1] / y, 1), order=3)
    
    input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 模型預測
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input_tensor), dim=1), dim=1).squeeze(0)
        pred = out.cpu().detach().numpy()

    # 調整分割結果大小為原始影像大小
    if x != patch_size[0] or y != patch_size[1]:
        pred = zoom(pred, (x / patch_size[0], y / patch_size[1]), order=0)

    # 疊加分割結果在原始影像上
    overlaid_image = overlay_segmentation_on_image(img, pred, alpha=0.5)

    # 顯示或儲存結果
    cv2.imshow('Segmented Overlay', overlaid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


