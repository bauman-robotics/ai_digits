#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
import matplotlib.pyplot as plt
#-----------------
from tensorflow import keras
from tensorflow.keras import layers
#-----------------
import numpy as np
from PIL import Image 
#-----------------
import os
#--- create model --------------
model = keras.Sequential([
layers.Dense(512, activation="relu"),
layers.Dense(10, activation="softmax")
])
#----- compile model -----------
model.compile(optimizer="rmsprop",
loss="sparse_categorical_crossentropy",
metrics=["accuracy"])
#-------------------------------

#-----------------------------------------------
# Восстановление состояния модели
path = os.getcwd()
model_file = path + '/my_model.keras'
print("model_file_name=", model_file)
model = keras.models.load_model(model_file)
#-----------------------------------------------
model.summary()


# In[14]:


type(train_images)
print('dtype =', train_images.dtype)
print('ndim =', train_images.ndim)
print('shape =', train_images.shape)

#plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.imshow(train_images[0], cmap=plt.cm.gray)

plt.show()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


# In[15]:


model.fit(train_images, train_labels, epochs=5, batch_size=128)
#model.fit(train_images, train_labels, epochs=100, batch_size=500)


# In[16]:


test_digits = test_images[0:10]
predictions = model.predict(test_digits)
#predictions[0]

print('#######################')
p_num = 0
print('p_num = ', p_num)
print('test_label -' , test_labels[p_num])
for i in range (10):
      print('is ', i, '---', "%.3f" %(predictions[p_num][i])) 
print('#######################')

p_num = 1
print('p_num = ', p_num)
print('test_label -' , test_labels[p_num])
for i in range (10):
      print('is ', i, '---', "%.3f" %(predictions[p_num][i])) 
print('#######################')


# In[140]:


# import numpy as np
# from PIL import Image 
# import matplotlib.pyplot as plt 
#import ai_functions as func

img00 = np.asarray(Image.open('./Test_imgs/0_ok.png').convert('L'))
img01 = np.asarray(Image.open('./Test_imgs/1_ok_1.png').convert('L'))
img02 = np.asarray(Image.open('./Test_imgs/2_ok_1.png').convert('L'))
img03 = np.asarray(Image.open('./Test_imgs/3_ok.png').convert('L'))
img04 = np.asarray(Image.open('./Test_imgs/4_ok.png').convert('L'))

img05 = np.asarray(Image.open('./Test_imgs/5_ok.png').convert('L'))
img06 = np.asarray(Image.open('./Test_imgs/6_ok_1.png').convert('L'))
img07 = np.asarray(Image.open('./Test_imgs/7_ok_1.png').convert('L'))
img08 = np.asarray(Image.open('./Test_imgs/8_ok_1.png').convert('L'))
img09 = np.asarray(Image.open('./Test_imgs/9_ok.png').convert('L'))

img00 = np.invert(img00)
img01 = np.invert(img01)
img02 = np.invert(img02)
img03 = np.invert(img03)
img04 = np.invert(img04)

img05 = np.invert(img05)
img06 = np.invert(img06)
img07 = np.invert(img07)
img08 = np.invert(img08)
img09 = np.invert(img09)

### plot ################
fig, axes = plt.subplots(2, 5, figsize=(6, 3))
ax = axes.ravel()

ax[0].imshow(img00, cmap=plt.cm.gray)
ax[0].set_title("img00")
##################
ax[1].imshow(img01, cmap=plt.cm.gray)
ax[1].set_title("img01")
##################
ax[2].imshow(img02, cmap=plt.cm.gray)
ax[2].set_title("img02")
##################
ax[3].imshow(img03, cmap=plt.cm.gray)
ax[3].set_title("img03")
##################
ax[4].imshow(img04, cmap=plt.cm.gray)
ax[4].set_title("img04")
##################

ax[5].imshow(img05, cmap=plt.cm.gray)
ax[5].set_title("img05")
##################
ax[6].imshow(img06, cmap=plt.cm.gray)
ax[6].set_title("img06")
##################
ax[7].imshow(img07, cmap=plt.cm.gray)
ax[7].set_title("img07")
##################
ax[8].imshow(img08, cmap=plt.cm.gray)
ax[8].set_title("img08")
##################
ax[9].imshow(img09, cmap=plt.cm.gray)
ax[9].set_title("img09")
##################


fig.tight_layout()
plt.show()
#######################

fig.savefig('plt.png')

#print('img01.dim =', img01.ndim)
#print('img01.shape =', img01.shape)
#print('img01[0][0] =', img01[0][0])


# In[18]:


img0 = img00.astype('float32') / 255
img1 = img01.astype('float32') / 255
img2 = img02.astype('float32') / 255
img3 = img03.astype('float32') / 255
img4 = img04.astype('float32') / 255
img5 = img05.astype('float32') / 255
img6 = img06.astype('float32') / 255
img7 = img07.astype('float32') / 255
img8 = img08.astype('float32') / 255
img9 = img09.astype('float32') / 255

img_arr = np.array( [ img0, img1, img2, img3, img4, img5, img6, img7, img8, img9  ] )

print('img_arr.dim =', img_arr.ndim) 
print('img_arr.shape =', img_arr.shape) 
#print('img_arr[0][2] =', img_arr[0][2])

img_arr = img_arr.reshape((10, 28 * 28)) 
print('####### Reshape #######')

print('img_array.dim =', img_arr.ndim) 
print('img_array.shape =', img_arr.shape) 
print('#######################')

print('####### Part1 Reshape Back #######') 
img_arr_part1 = img_arr[8].reshape((28 , 28)) 
print('img_arr_part1.dim =', img_arr_part1.ndim) 
print('img_arr_part1.shape =', img_arr_part1.shape) 
#plt.imshow(img_arr_part1, cmap=plt.cm.gray) 
print('#######################')


# In[19]:


predictions = model.predict(img_arr)
#predictions[0]

print('#######################')
for p_num in range (10):
    print('p_num = ', p_num)
    for i in range (10):
          print('is ', i, '---', "%.3f" %(predictions[p_num][i]))  
    print('#######################')

## 7, 8, 9


# In[20]:


# Сохранение весов в файл HDF5
model.save(model_file)
#model.save_weights('my_model.keras')


# In[21]:





# In[114]:


######################################################
######################################################
### Make folders for Test Images for work in server 
import shutil
work_directory = path
input_directory = str(work_directory) + "/web_test_imgs"
result_directory = str(work_directory) + "/web_results" 
result_file_name = "/result.c"
test_str = "Hello"
file_h = 28
file_w = 28
#MAX_COUNT_TEST_IMGS = 100
#-------------------------------
try:   
    os.mkdir(input_directory)
except OSError as error:
    do_nothing = 1
#    shutil.rmtree(input_directory)
#    os.mkdir(input_directory)
#-------------------------------
try:   
    os.mkdir(result_directory)
except OSError as error:
    shutil.rmtree(result_directory)
    os.mkdir(result_directory)
#-------------------------------


res_flie = open(result_directory + result_file_name, mode="w")
res_flie.write(str( test_str   +'\n'))    
res_flie.close()


# In[115]:


#=== Get list of test files 
test_files = os.listdir(input_directory)
num_test_files = len(test_files)
info_Count_files = "Count_files - " + str(num_test_files)
first_file_name = "First_file_name - " + test_files[0]
file_count = 0

res_flie = open(result_directory + result_file_name, mode="w")
res_flie.write(info_Count_files  +'\n')    
res_flie.write(first_file_name   +'\n') 
res_flie.close()


# In[125]:


# Try to find test image File in folder "./web_test_imgs" 
MAX_COUNT_TEST_IMGS = 100

#=== Check if it right test_files 
f_name_ok  = ['' for _ in range(MAX_COUNT_TEST_IMGS)]
f_name_bad = ['' for _ in range(MAX_COUNT_TEST_IMGS)]
file_ok_count = 0
file_bad_count = 0
for file_count in range(num_test_files):
    test_img = np.asarray(Image.open(f_name[file_count]).convert('L'))
    file_shape = test_img.shape
    file_dim = test_img.ndim
    if ((file_shape[0] == 28) and (file_shape[1] == 28) and (file_dim == 2)) :
        f_name_ok[file_ok_count] = test_files[file_count];
        file_ok_count = file_ok_count + 1
    else :
        f_name_bad[file_bad_count] = test_files[file_count];
        file_bad_count = file_bad_count + 1


# In[ ]:





# In[136]:


#=== Check Ok file list 
res_flie = open(result_directory + result_file_name, mode="w")

res_flie.write("Warning: MAX_COUNT_TEST_IMGS = " + str(MAX_COUNT_TEST_IMGS)  +'\n')
ok_list = "#=== Ok file list =========="
ok_num = str(len(f_name_ok))
res_flie.write(ok_list   +'\n')
res_flie.write("count = " + str(file_ok_count)   +'\n')
for file_count in range(file_ok_count):
    ok_name = str(file_count) + "   " + f_name_ok[file_count]
    res_flie.write(ok_name   +'\n')
res_flie.write("#============================"   +'\n') 


# In[137]:


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


# In[ ]:





# In[138]:


get_ipython().system('jupyter nbconvert --to script digits_v01.ipynb')


# In[ ]:




