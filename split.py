from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pdb

ret_list=[]
with open('/Users/anushaprakash/Desktop/all_full_dev.lst',"r") as f:
    all_lines=f.readlines()
    for now_line in all_lines:
        tmp_str=now_line.split()
        vid_name=tmp_str[0];vid_label=tmp_str[1]
        ret_list.append((vid_name,vid_label))

    print(len(ret_list))

X = [a for (a,b)  in ret_list]
Y = [b for (a,b)  in ret_list]

X = np.array(X)
Y = np.array(Y)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3333)
sss.get_n_splits(X, Y)

for train_index, test_index in sss.split(X, Y):
	print(len(train_index), len(test_index))
	
	X_train = X[train_index].tolist()
	X_test = X[test_index].tolist()
	
	Y_train = Y[train_index].tolist()
	Y_test = Y[test_index].tolist()

	with open('/Users/anushaprakash/Desktop/train.lst',"w") as f:
		for i in range(len(X_train)):
			f.write(X_train[i]+" "+Y_train[i])
			f.write("\n")
	f.close()
	with open('/Users/anushaprakash/Desktop/test.lst',"w") as f:
                for i in range(len(X_test)):
                        f.write(X_test[i]+" "+Y_test[i])
                        f.write("\n")
	f.close()	
