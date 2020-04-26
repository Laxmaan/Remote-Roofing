#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

r"""Generate tracking results for videos using Siamese Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import os.path as osp
import sys
from glob import glob
import json
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sacred import Experiment
import numpy as np
CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from inference import inference_wrapper
from inference.tracker import Tracker
from utils.infer_utils import Rectangle
from utils.misc_utils import auto_select_gpu, mkdir_p, sort_nicely, load_cfgs
from PIL import Image
ex = Experiment()


@ex.config
def configs():
  checkpoint = 'Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained'
  input_files = 'assets/TwitterCV'


@ex.automain
def main(checkpoint, input_files):
  os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()

  model_config, _, track_config = load_cfgs(checkpoint)
  track_config['log_level'] = 1

  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(model_config, track_config, checkpoint)
  g.finalize()

  if not osp.isdir(track_config['log_dir']):
    logging.info('Creating inference directory: %s', track_config['log_dir'])
    mkdir_p(track_config['log_dir'])

  video_dirs = []
  for file_pattern in input_files.split(","):
    video_dirs.extend(glob(file_pattern))
  logging.info("Running tracking on %d videos matching %s", len(video_dirs), input_files)

  gpu_options = tf.GPUOptions(allow_growth=True)
  sess_config = tf.ConfigProto(gpu_options=gpu_options)

  with tf.Session(graph=g, config=sess_config) as sess:
    restore_fn(sess)

    tracker = Tracker(model, model_config=model_config, track_config=track_config)

    for video_dir in video_dirs:
      if not osp.isdir(video_dir):
        logging.warning('{} is not a directory, skipping...'.format(video_dir))
        continue

      video_name = osp.basename(video_dir)
      video_log_dir = osp.join(track_config['log_dir'], video_name)
      mkdir_p(video_log_dir)

      filenames = sort_nicely(glob(video_dir + '/img/*.jpg')+glob(video_dir + '/img/*.png'))
      first_line = open(video_dir + '/groundtruth_rect.txt').readline()
      bb = [int(v) for v in first_line.strip().split(',')]
      init_bb = Rectangle(bb[0] - 1, bb[1] - 1, bb[2], bb[3])  # 0-index in python

      trajectory = tracker.track(sess, init_bb, filenames, video_log_dir)
      with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
        for region in trajectory:
          rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
                                            region.width, region.height)
          f.write(rect_str)

      with open(osp.join(video_log_dir,'bboxes.json'),'r') as f:
        data = json.load(f)

      final_output={}
      for i,fname in enumerate(data.keys()):
        img = np.array(Image.open(fname).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #print(img,img.shape)
        bboxes = data[fname]
        bboxes = list(map(lambda x: list(map(lambda y: float(y), x.strip().split(','))),bboxes))
        arr=[]
        for x,y,w,h in bboxes:
          ymin,xmin,ymax,xmax = int(y),int(x),int(y+h),int(x+w)
          img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0,255),2)
          arr.append([ymin,xmin,ymax,xmax])
        final_output[fname]=arr
        name =osp.basename(fname)
        name =osp.splitext(name)[0]

        W,H,_ = img.shape
        cv2.imshow("Pic",cv2.resize(img,(W//2,H//2)))
        cv2.waitKey(0)

        out_folder = osp.join(video_log_dir,"Outputs")
        mkdir_p(out_folder)
        cv2.imwrite(osp.join(out_folder,f"{name}_bbox.png"),img)

      with open(osp.join(out_folder,"output.json"),"w") as f:
        json.dump(final_output,f,indent=4)
      
      cv2.destroyAllWindows()

        

