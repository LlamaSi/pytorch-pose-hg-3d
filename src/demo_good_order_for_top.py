from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import cv2
import numpy as np
import torch
import torch.utils.data
from opts import opts
from model import create_model
from utils.debugger import Debugger
from utils.image import get_affine_transform, transform_preds
from utils.eval import get_preds, get_preds_3d
from tqdm import tqdm

import pdb
image_ext = ['jpg', 'jpeg', 'png']
mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
corres = [10,9,8,11,12,13,0,0,1,0,4,3,2,5,6,7]

def is_image(file_name):
  ext = file_name[file_name.rfind('.') + 1:].lower()
  return ext in image_ext


def demo_image(image, model, opt, name):
  s = max(image.shape[0], image.shape[1]) * 1.0
  c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
  trans_input = get_affine_transform(
      c, s, 0, [opt.input_w, opt.input_h])
  inp = cv2.warpAffine(image, trans_input, (opt.input_w, opt.input_h),
                         flags=cv2.INTER_LINEAR)
  inp = (inp / 255. - mean) / std
  inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
  inp = torch.from_numpy(inp).to(opt.device)
  out = model(inp)[-1]

  pred = get_preds(out['hm'].detach().cpu().numpy())[0]
  pred = transform_preds(pred, c, s, (opt.output_w, opt.output_h))
  # pred 2d range (176, 256)
  pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(), 
                         out['depth'].detach().cpu().numpy())[0]
  pred_3d_real_size = pred_3d*4
  pred_3d_real_size[:, 0] = pred_3d_real_size[:, 0] - 40
  # print(pred_3d)
  # pdb.set_trace()
  pred_3d_ordered = np.zeros([15,3])
  # the last one as mid hip for spline compute
  for i in range(16):
    pred_3d_ordered[corres[i]] = pred_3d_real_size[i]

  pred_3d_ordered[1] = (pred_3d_ordered[2] + pred_3d_ordered [5]) / 2
  pred_3d_ordered[14] = (pred_3d_ordered[8] + pred_3d_ordered [11]) / 2

  pred_3d_ordered[0] = -1
  pred_3d_ordered[9:11] = -1
  pred_3d_ordered[12:14] = -1

  from good_order_cood_angle_convert import absolute_angles, anglelimbtoxyz2
  # bias
  # neck as the offset
  # if pred_3d[8,:][0] != 0 or pred_3d[8,:][1] != 0:

  # bias = np.array([pred[8,0], pred[8,1]])
  absolute_angles, limbs, offset = absolute_angles(pred_3d_ordered)
  # pdb.set_trace()
  # rev = anglelimbtoxyz2(offset, absolute_angles, limbs)
  pred_2d = pred_3d_ordered[:,:2]

  dic = {'absolute_angles': absolute_angles, 'limbs':limbs, 'offset': offset}
  # pdb.set_trace()
  np.save(name, dic)
  # print(name)
  # debugger = Debugger()
  # debugger.add_img(image)
  # debugger.add_point_2d(pred, (255, 0, 0))
  # debugger.add_point_3d(pred_3d, 'b')
  # debugger.show_all_imgs(pause=False)
  # debugger.show_3d()

def main(opt):
  opt.heads['depth'] = opt.num_output
  if opt.load_model == '':
    opt.load_model = '../models/fusion_3d_var.pth'
  if opt.gpus[0] >= 0:
    opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
  else:
    opt.device = torch.device('cpu')
  
  model, _, _ = create_model(opt)
  model = model.to(opt.device)
  model.eval()
  # pdb.set_trace()
  if os.path.isdir(opt.demo):
    ls = os.listdir(opt.demo)
    for file_name in tqdm(sorted(ls)):
      if is_image(file_name):
        image_name = os.path.join(opt.demo, file_name)
        # print('Running {} ...'.format(image_name))
        image = cv2.imread(image_name)
        folder3d = opt.demo.rstrip('/') + '_3d_top_ordered'
        name = os.path.join(folder3d, file_name)
        demo_image(image, model, opt, name.replace('.jpg', '.npy'))
  elif is_image(opt.demo):
    print('Running {} ...'.format(opt.demo))
    image = cv2.imread(opt.demo)
    demo_image(image, model, opt, name)
    

if __name__ == '__main__':
  opt = opts().parse()
  data_root = '/home/wenwens/Documents/HumanPose/Pose-Transfer-vae/fashion_data'
  splits = ['train', 'test']
  for split in splits:
    split_path = os.path.join(data_root, split)
    opt.demo = split_path
    main(opt)
