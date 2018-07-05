import cv2
import os
import time
import numpy as np
import sys
sys.path.append("../")
import utils
import pdb
import configs.hw2_config as config

def extract_surf():
	k = 256
	all_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + \
                           utils.get_video_and_label_list(config.all_test_list_filename) + \
                           utils.get_video_and_label_list(config.all_val_list_filename)

	for now_video_label in all_video_label_list:
		vid_name = now_video_label[0]
		#print(vid_name)
		ds_file=os.path.join(config.ds_video_root_path,vid_name+config.ds_video_file_format)
	        ds_file_video = cv2.VideoCapture(ds_file)
		
		key_frame = []
	
		length = int(ds_file_video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
		#print(length)
		c = 0
		while(ds_file_video.isOpened()):

			ret, frame = ds_file_video.read(0)
			
			if ret != True :
				break
			if (c%5 == 0):
				frame = frame.astype(np.uint8)	
				try:
					surf = cv2.SURF(400)
					keypoints, descriptors = surf.detectAndCompute(frame,None)
					key_frame.append(descriptors)
					print(c,len(keypoints),descriptors.shape)
				except:
					print("No",vid_name)
			c = c+1
		ds_file_video.release()
			#print(frame)
			#print(frame.shape)
		#print(len(key_frame))	
		
		#print(c)
		np.save(os.path.join(config.surf_feat_path,vid_name+config.surf_feat_file_format),key_frame)
		'''
		i = 0
		while(ds_file_video.isOpened()):
			
		     if i % 5 == 0:
			ret, frame = ds_file_video.read(0)
			#if (frame == None):
			#	continue
			try:
				gray = frame.astype(np.uint8)
			except:
				continue
			#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			surf = cv2.SURF()
			keypoints, descriptors = surf.detectAndCompute(gray,None)
	 		print(i,descriptors.shape)
			key_frame.append(descriptors)
		        #kmeans_input.append(descriptors)

  		     i = i+

		np.save(vid_name+config.surf_feat_file_format,key_frame)
		
		#key_frame_video.append(key_frame)
		end = time.time()
		print(i,"Time",end-start)
		'''
	#kmeans_input = np.vstack(kmeans_input)
	#print(kmeans_input.shape)

	#kmeans = KMeans(n_clusters=k, random_state=0).fit(kmeans_input)	
	#k_means_clusters = kmeans.cluster_centers_
	#k_means_path = os.path.join(config.dataset_root_path,"kmeans_"+str(k)+"_surf_clusters.npy")
	#np.save(k_means_path, k_means_clusters)

	
if __name__== "__main__":
	extract_surf()
