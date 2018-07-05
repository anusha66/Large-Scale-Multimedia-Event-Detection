from sklearn.cluster import KMeans
import random
import pdb
import os
import sys
sys.path.append("../")
import utils
import numpy as np

def soundnet_bow(config,conv,size,k):

	all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
			       utils.get_video_and_label_list(config.all_val_list_filename)
	
	i =1

	for now_video_label in all_video_label_list:

            vid_name = now_video_label[0]
            sn_filename=os.path.join(config.soundnet_root_path,vid_name+conv+config.soundnet_file_format)

            if os.path.isfile(sn_filename):

                sn = np.load(sn_filename)
                sn = sn["arr_0"]
                sn = sn.reshape(-1, sn.shape[-1])
		
		sn = sn.T
		#index = int(np.floor(0.20*sn.shape[1]))

                #start = 0
                #end = sn.shape[1]

                #cols = random.sample(range(start, end), index)
                #sn = sn[:,cols]
                
		if i == 1 :
                        sn_vec = sn.T
                else :
		        sn_vec = np.concatenate((sn_vec,sn.T),axis=0)
                i = i + 1
	
   	kmeans = KMeans(n_clusters=k, random_state=0).fit(sn_vec)
	k_means_clusters = kmeans.cluster_centers_
  	k_means_path = os.path.join(config.cluster_classifiers,"kmeans_"+str(k)+""+str(conv)+"_sn_clusters.npy")

    	np.save(k_means_path, k_means_clusters)
	
	all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
                           utils.get_video_and_label_list(config.all_val_list_filename) + \
                           utils.get_video_and_label_list(config.all_test_list_filename)

        k_means_path = os.path.join(config.cluster_classifiers,"kmeans_"+str(k)+""+str(conv)+"_sn_clusters.npy")

        k_means_clusters = np.load(k_means_path)	

	for now_video_label in all_video_label_list:
		
		vid_name = now_video_label[0]
		soundnet_filename = os.path.join(config.soundnet_root_path,vid_name+conv+config.soundnet_file_format) 
		fea = os.path.join(config.soundnet_fea_bow,vid_name+conv+config.soundnet_fea_file_format)  
		if os.path.isfile(soundnet_filename):	
			sn = np.load(soundnet_filename)
                        sn = sn["arr_0"]
			sn = sn.reshape(-1, sn.shape[-1])
                	k_dim = np.zeros((1,k))

			for j in range(sn.shape[0]):

                        	index = np.argmin(np.linalg.norm(sn[j,:] - k_means_clusters,axis=1))

                        	k_dim[0][int(index)] = k_dim[0][int(index)] + 1

                       	np.save(fea,k_dim)

		else:
			np.save(fea,np.zeros((1,k)))		
			

		
