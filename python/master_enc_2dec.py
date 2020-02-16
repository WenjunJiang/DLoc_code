import os
from shutil import copyfile
# from subprocess import call
import time

experiments = ["e19"]
sequence_of_exec = ["real2dec_train","real2dec_test"]
#  ,"real2dec_disjoint_test"
#  					"real2dec_test_1env","real2dec_test_3env","real2dec_test_4env"]
USE_PYTHON3_EXPLICIT = True
params_dir = "./params/"

for i, e in enumerate(experiments):

	for s in sequence_of_exec:
		file_name = e + "_" + s + ".py"
		src = params_dir+file_name
		dst = "./params.py"
		if not os.path.isfile(src):
			continue
		# print(src, dst)
		copyfile(src, dst)
		print("copied params, calling main on them")
		if(USE_PYTHON3_EXPLICIT):
			print("Using Explicit call to python3")
			os.system('python3 main_enc_2dec.py')
		else:
			os.system('python main_enc_2dec.py')
		time.sleep(10)