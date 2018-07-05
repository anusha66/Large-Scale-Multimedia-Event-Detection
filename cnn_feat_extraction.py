from keras.applications import VGG19
from moviepy.editor import VideoFileClip
import numpy as np
import argparse
import time
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
import os
import sys
sys.path.append("../")
import utils
import configs.hw2_config as config
import pdb
import scipy.misc

if __name__ == '__main__':

	shape = (224, 224)
	mod = VGG19(weights='imagenet')
	model = Model(inputs=mod.input,outputs=mod.layers[-1].output)

	all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
                           utils.get_video_and_label_list(config.all_test_list_filename) + \
                           utils.get_video_and_label_list(config.all_val_list_filename)

        for now_video_label in all_video_label_list:
		vid_name = now_video_label[0]
                ds_file=os.path.join(config.ds_video_root_path,vid_name+config.ds_video_file_format)
                cnn_file=os.path.join(config.cnn_feat_path,vid_name+config.cnn_feat_file_format)
		if(os.path.exists(cnn_file)):
			continue
		#print(vid_name)
		clip = VideoFileClip(ds_file)
		#frames = [idx for idx, x in enumerate(clip.iter_frames()) if idx % 5 == 0]
		#pdb.set_trace()
		frames = [scipy.misc.imresize(x, shape) for idx, x in enumerate(clip.iter_frames()) if idx % 50 == 0]
		#print(len(frames))
		
                
		ini = 1
		for fr in frames:
			fr = np.array(fr, dtype=np.float64)
			
			fr = np.expand_dims(fr, axis=0)
			fr = preprocess_input(fr)
			
			features = model.predict(fr)
			
			if ini == 1:
				vid_features = features
				ini = 0
			else:
				vid_features = np.vstack((vid_features,features))
		fin_vid_fea = np.mean(vid_features, axis=0)
		fin_vid_fea = np.reshape(fin_vid_fea,(1,fin_vid_fea.shape[0]))
		np.save(cnn_file,fin_vid_fea)
