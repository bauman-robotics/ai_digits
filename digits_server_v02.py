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


# In[2]:


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


# In[3]:


#model.fit(train_images, train_labels, epochs=5, batch_size=128)  # work mode

####model.fit(train_images, train_labels, epochs=100, batch_size=500)  

# Сохранение весов в файл HDF5
model.save(model_file)
#model.save_weights('my_model.keras')


# In[4]:


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


# In[5]:


#################################################################################
#################################################################################
#################################################################################
#=== Get Test Imgs Names and Valid Count ===
import Test_Folder_Packet
import os           # temp 
path = os.getcwd()  # temp
f_name_ok  = ['' for _ in range(Test_Folder_Packet.MAX_COUNT_TEST_IMGS)]

f_ok_count = Test_Folder_Packet.Test_Imgs_Get(path, f_name_ok)

#print(str(Test_Folder_Packet.MAX_COUNT_TEST_IMGS))
print("count right test_files = ", str(f_ok_count))
#print("first_file_name  = ", f_name_ok[0])
#print("second_file_name = ", f_name_ok[1])


# In[6]:


#=== Get Test Imgs ===
test_imgs  = ['' for _ in range(f_ok_count)]
imgs  = np.zeros((f_ok_count,28,28))
for i in range(f_ok_count) :
    test_imgs[i] = np.asarray(Image.open(path + "/web_test_imgs/" + f_name_ok[i]).convert('L'))
    imgs[i] = np.invert(test_imgs[i])
    imgs[i] = imgs[i].astype('float32') / 255


# In[7]:


#=== plot Test Imgs ===

plt_w_count = 5
plt_h_count = 2
fig, axes = plt.subplots(plt_h_count, plt_w_count, figsize=(6, 3))
ax = axes.ravel()

if (f_ok_count > 10) :
    plot_count = 10
else :
    plot_count = f_ok_count
    
for i in range(plot_count) :    
    ax[i].imshow(imgs[i], cmap=plt.cm.gray)
    ax[i].set_title(str(f_name_ok[i]))

fig.tight_layout()
plt.show()
#=== save plot Test Imgs ===

fig.savefig('plt.png')


# In[8]:


#=== Reshape ====
print('imgs.dim =', imgs[0].ndim) 
print('imgs.shape =', imgs[0].shape) 
imgs = imgs.reshape((f_ok_count, Test_Folder_Packet.file_h * Test_Folder_Packet.file_w)) 
print('test_imgs.dim =', imgs[0].ndim) 
print('test_imgs.shape =', imgs.shape) 


# In[9]:


predictions = model.predict(imgs)
#predictions[0]

print('#######################')
for p_num in range (10):
    print('img_name = ', f_name_ok[p_num])
    for i in range (10):
          print('is ', i, '---', "%.3f" %(predictions[p_num][i]))  
    print('#######################')


# In[10]:


#=== Save results to file ===
import Test_Folder_Packet
Test_Folder_Packet.Save_Result(predictions)


# In[11]:


get_ipython().system('jupyter nbconvert --to script digits_server_v02.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




