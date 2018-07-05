import random
import os
import time
import numpy as np
import sys
sys.path.append("../")
import utils
import pdb
from sklearn.cluster import KMeans
import configs.hw2_config as config


def kmeans_surf(k):

	all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
 			utils.get_video_and_label_list(config.all_val_list_filename)
                           #utils.get_video_and_label_list(config.all_val_list_filename)
	i = 0

	for now_video_label in all_video_label_list:
		vid_name = now_video_label[0]
		print(vid_name)
		vid_kf=os.path.join(config.surf_feat_path,vid_name+config.surf_feat_file_format)
		vid_kf=np.load(vid_kf)	

		new_vid_kf = []
		
		for each in vid_kf:
							
				index = int(np.ceil(0.02*each.shape[0]))
				
				start = 0
                		end = each.shape[0]
			
				rows = random.sample(range(start, end), index)
				each = each[rows,]
				new_vid_kf.append(each)
		
		vid_kf = new_vid_kf	
		
		if i == 0:	
			new = np.vstack(vid_kf)
			i = 1
		else:
			temp = np.vstack(vid_kf)
			new = np.vstack((new,temp))
		
	
	kmeans_input = new
	kmeans = KMeans(n_clusters=k, random_state=0).fit(kmeans_input)
	k_means_clusters = kmeans.cluster_centers_
        k_means_path = os.path.join(config.dataset_root_path,"kmeans_"+str(k)+"_surf_clusters.npy")
        np.save(k_means_path, k_means_clusters)

def vlad_surf(k):

	k_means_path = os.path.join(config.dataset_root_path,"kmeans_"+str(k)+"_surf_clusters.npy")
        k_means_clusters = np.load(k_means_path)

        all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
                           utils.get_video_and_label_list(config.all_test_list_filename) + \
                           utils.get_video_and_label_list(config.all_val_list_filename)

        for now_video_label in all_video_label_list:
                vid_name = now_video_label[0]
                vid_kf=os.path.join(config.surf_feat_path,vid_name+config.surf_feat_file_format)
                vid_kf = np.load(vid_kf)
                k_dim = np.zeros((len(vid_kf),k))
                surf_path=os.path.join(config.surf_vlad_path,vid_name+config.surf_vlad_file_format)
                print(len(vid_kf))
                i = 0
                for kf in vid_kf:

			k_dim = np.zeros((k,128))

                        for j in range(kf.shape[0]):
				
                                index =  np.argmin(np.linalg.norm(kf[j,:] - k_means_clusters,axis=1))
                                diff = kf[j,:] - k_means_clusters[index]
			        k_dim[int(index)] = k_dim[int(index)]+diff

                        k_dim = k_dim.flatten()
			k_dim = k_dim.reshape((1,k_dim.shape[0]))

			norm = np.linalg.norm(k_dim)
                	k_dim = k_dim/norm
			
			if i == 0:
				k_dim_vid = k_dim
			else:
				k_dim_vid = np.vstack((k_dim_vid,k_dim))

			i = i+1

		k_dim_vid = np.mean(k_dim_vid,axis=0)
                k_dim_vid = k_dim_vid.reshape((1,k_dim_vid.shape[0]))
                np.save(surf_path,k_dim_vid)
			
if __name__== "__main__":
        kmeans_surf(350)
	vlad_surf(350)
