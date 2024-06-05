######################################################
######################################################
### Make folders for Test Images for work in server 
import shutil
import os
import numpy as np
from PIL import Image 


file_h = 28
file_w = 28
MAX_COUNT_TEST_IMGS = 100
#f_name_ok  = ['' for _ in range(MAX_COUNT_TEST_IMGS)]
#file_ok_count = 0
#-------------------------------
def Test_Imgs_Get(path, f_name_ok) :    
	#-------------------------------
	#file_ok_count = 0
	#-------------------------------
	f_name_bad = ['' for _ in range(MAX_COUNT_TEST_IMGS)]
	file_bad_count = 0
	#-------------------------------
	work_directory = path
	input_directory = str(work_directory) + "/web_test_imgs/"
	result_directory = str(work_directory) + "/web_results" 
	result_file_name = "/result.c"
	try:   
	    os.mkdir(input_directory)
	except OSError as error:
	    do_nothing = 1
	#-------------------------------
	try:   
	    os.mkdir(result_directory)
	except OSError as error:
	    shutil.rmtree(result_directory)
	    os.mkdir(result_directory)
	#-------------------------------
	#=============================================================
	#=== Get list of test files 
	test_files = os.listdir(input_directory)
	num_test_files = len(test_files)
	info_Count_files = "Count_files - " + str(num_test_files)
	first_file_name = "First_file_name - " + test_files[0]
	file_count = 0
	
	res_flie = open(result_directory + result_file_name, mode="w")
	res_flie.write(info_Count_files  +'\n')    
	res_flie.write(first_file_name   +'\n') 
	#res_flie.close()
	#=============================================================
	# Try to find test image File in folder "./web_test_imgs" 
	#=== Check if it right test_files 
	# f_name_ok  = ['' for _ in range(MAX_COUNT_TEST_IMGS)]
	# f_name_bad = ['' for _ in range(MAX_COUNT_TEST_IMGS)]
	file_ok_count = 0
	# file_bad_count = 0
	for file_count in range(num_test_files):
		test_img = np.asarray(Image.open(input_directory + test_files[file_count]).convert('L'))
		file_shape = test_img.shape
		file_dim = test_img.ndim
		if ((file_shape[0] == file_w) and (file_shape[1] == file_h) and (file_dim == 2)) :  # may be wrong w and h place 
			f_name_ok[file_ok_count] = test_files[file_count]
			file_ok_count = file_ok_count + 1
		else :
			f_name_bad[file_bad_count] = test_files[file_count];
			file_bad_count = file_bad_count + 1
	#=============================================================
	#=== Check Ok file list 
	#res_flie = open(result_directory + result_file_name, mode="w")
	    
	res_flie.write("Warning: MAX_COUNT_TEST_IMGS = " + str(MAX_COUNT_TEST_IMGS)  +'\n')
	ok_list = "#=== Ok file list =========="
	ok_num = str(len(f_name_ok))
	res_flie.write(ok_list   +'\n')
	res_flie.write("count = " + str(file_ok_count)   +'\n')
	for file_count in range(file_ok_count):
		ok_name = str(file_count) + "   " + f_name_ok[file_count]
		res_flie.write(ok_name   +'\n')
	res_flie.write("#============================"   +'\n')
	#=============================================================
	#=== Check OBad file list 
	#res_flie = open(result_directory + result_file_name, mode="w")
	bad_list = "#=== Bad file list =========="
	bad_num = str(len(f_name_bad))
	res_flie.write(bad_list   +'\n')
	res_flie.write("count = " + str(file_bad_count)   +'\n')
	for file_count in range(file_bad_count):
		bad_name = str(file_count) + "   " + f_name_bad[file_count]
		res_flie.write(bad_name   +'\n')
	res_flie.write("#============================"   +'\n') 
	res_flie.close()
	#=============================================================
	return file_ok_count