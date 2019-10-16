import os
import numpy as np
import cv2 
from skimage.transform import pyramid_reduce

###사용하려면 경로 맞게 수정할 것

data_path =  "./celeba-dataset"
img_path = os.path.join(data_path,"img_align_celeba/img_align_celeba")
pp_img_path = os.path.join("/media/elensar92/NTFS_SHARING","Training")

#print("img_path : ",img_path )

evaluation_list = np.loadtxt(os.path.join(data_path,"list_eval_partition.csv"), dtype=str, delimiter=',', skiprows=1)
##print(evaluation_list[0][0]) ## evaluation_list[idx][p] 이미지파일idx.jpg 의 정보 p==0 파일명, p==1 (0,1,2) 각각 훈련용,검증용,테스트용 

scale = 4 ## 4배로 줄일것임
train_num = 162770 ## 테스트 데이터 162770개
val_num = 19867 ## 발리데이션 데이터 19867개
test_num = 19962  ## 테스트 데이터 19962개

for i,v in enumerate(evaluation_list) :
	file_name,ext = os.path.splitext(v[0])
	print("file_name : ",file_name)
	new_img_path = os.path.join(img_path, v[0])

	img = cv2.imread(new_img_path)
	#print("img shape : ",img.shape)
	height,width,_ = img.shape

	crop_img = img[int((height-width)//2):int(-(height-width)//2), :] ##이미지 중앙부만 남기고 버림
	crop_img = cv2.resize(crop_img, dsize=(176,176))

	resized_img = pyramid_reduce(crop_img,downscale=scale) ## 1/4로 줄임
	normalized_img = cv2.normalize(crop_img.astype(np.float64),None,0,1,cv2.NORM_MINMAX)

	###### csv 에서 읽어들인 데이터 인덱스에 따라 나눔 ######
	##print(os.path.join(pp_img_path,"x_train",file_name+'.npy'))
	if v[1] == '0' :
		np.save(os.path.join(pp_img_path,"x_train",file_name+'.npy'),resized_img)
		np.save(os.path.join(pp_img_path,"y_train",file_name+'.npy'),normalized_img)
	elif v[1] == '1' :
		np.save(os.path.join(pp_img_path,"x_valid",file_name+'.npy'),resized_img)
		np.save(os.path.join(pp_img_path,"y_valid",file_name+'.npy'),normalized_img)
	elif v[1] == '2':
		np.save(os.path.join(pp_img_path,"x_test",file_name+'.npy'),resized_img)
		np.save(os.path.join(pp_img_path,"y_test",file_name+'.npy'),normalized_img)

	#if cv2.waitKey(0) == ord('q') :
	#	break


