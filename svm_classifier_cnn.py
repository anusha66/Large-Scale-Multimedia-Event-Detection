import sys
sys.path.append("../")
import os
import utils as utils
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import pickle
import pdb
import configs.hw2_config as config
# TODO: implement the SVM classifier training function here.
# it should load different encoded videos features (MFCC, ASR, etc.) in all_trn.lst (validation)
# or all_trn.lst+all_val.lst (final submission) and train corresponding classifiers for each event types.

def train(k):

    X = np.zeros((1,k))
    
    Y_1 = []
    Y_2 = []
    Y_3 = []

    train_video_label_list = utils.get_video_and_label_list(config.all_train_list_filename) + utils.get_video_and_label_list(config.all_val_list_filename)
  	
    for now_video_label in train_video_label_list: 

	vid_name = now_video_label[0]
	vid_label = now_video_label[1]
	cnn_feature_file=os.path.join(config.cnn_feat_path,vid_name+config.cnn_feat_file_format)

	if os.path.isfile(cnn_feature_file):
		
		cnn_feature = np.load(cnn_feature_file)
		X = np.vstack([X, cnn_feature])
       		
		if vid_label == "P001":
			Y_1.extend([1])
		else:
			Y_1.extend([0])

		
		if vid_label == "P002":
                        Y_2.extend([1])
                else:
                        Y_2.extend([0])

		
		if vid_label == "P003":
                        Y_3.extend([1])
                else:
                        Y_3.extend([0])

    X = X[:][1:]
   
    tuned_parameters = {'C': [0.001,0.01,0.1,1.0,10.0,100.0]}

    svm_clf_1 = LinearSVC()
    svm_clf_1.fit(X, Y_1)

    svm_clf_2 = LinearSVC()
    svm_clf_2.fit(X, Y_2)

    svm_clf_3 = LinearSVC(C=100)
    svm_clf_3.fit(X, Y_3)


    with open(os.path.join(config.cluster_classifiers,'svm1_cnn.pkl'), 'wb') as f1:
   	pickle.dump(svm_clf_1, f1)

    with open(os.path.join(config.cluster_classifiers,'svm2_cnn.pkl'), 'wb') as f2:
        pickle.dump(svm_clf_2, f2)

    with open(os.path.join(config.cluster_classifiers,'svm3_cnn.pkl'), 'wb') as f3:
        pickle.dump(svm_clf_3, f3)
		

# TODO: implement the SVM classifier testing (predicting) function here
# it should load classifiers of each event-feature combination and output prediction score
# for each video to a score file. For example, to submit results of videos in all_tst_fake.lst with MFCC feature
# using P001 event classifier. The test function should finally output the score file as P001_mfcc.lst. This is
# exactly what you should include under the "scores/" path in your ANDREWID_HW1.zip submission.

def test(k):
    	
    with open(os.path.join(config.cluster_classifiers,'svm1_cnn.pkl'), 'rb') as f1:
        	svm_clf_1 = pickle.load(f1)

    with open(os.path.join(config.cluster_classifiers,'svm2_cnn.pkl'), 'rb') as f2:
		svm_clf_2 = pickle.load(f2)
	
    with open(os.path.join(config.cluster_classifiers,'svm3_cnn.pkl'), 'rb') as f3:
		svm_clf_3 = pickle.load(f3)
	
    X = np.zeros((1,k))
    
    Y_1 = []
    Y_2 = []
    Y_3 = []

    with open(os.path.join(config.score,'gt_cnn.lst'),'wb') as f:
    	val_video_label_list = utils.get_video_and_label_list(config.all_test_list_filename)
	i =0
    	for now_video_label in val_video_label_list:
        	vid_name = now_video_label[0]
        	vid_label = now_video_label[1]

		cnn_feature_file=os.path.join(config.cnn_feat_path,vid_name+config.cnn_feat_file_format)
        	if os.path.isfile(cnn_feature_file):
			#print(i,vid_name)		
                	cnn_feature = np.load(cnn_feature_file)
               		X = np.vstack([X, cnn_feature])
			i=i+1	
			f.write(now_video_label[0]+" "+now_video_label[1])	
			f.write("\n")	
    f.close()
    X = X[:][1:]
    #X = X.astype(np.float)
    Y_1 = np.array(Y_1)
    Y_2 = np.array(Y_2)
    Y_3 = np.array(Y_3)
    
    svm_predicted_1 = svm_clf_1.decision_function(X)
    svm_predicted_2 = svm_clf_2.decision_function(X)
    svm_predicted_3 = svm_clf_3.decision_function(X)

    np.savetxt(os.path.join(config.score,"P001_cnn.lst"),svm_predicted_1)
    np.savetxt(os.path.join(config.score,"P002_cnn.lst"),svm_predicted_2)
    np.savetxt(os.path.join(config.score,"P003_cnn.lst"),svm_predicted_3)

if __name__=="__main__" :  
	 pass
