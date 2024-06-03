#!/usr/bin/env python
# coding: utf-8

# In[31]:


from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
import matplotlib.pyplot as plt


# In[32]:


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


# In[33]:


from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
layers.Dense(512, activation="relu"),
layers.Dense(10, activation="softmax")
])


# In[34]:


model.compile(optimizer="rmsprop",
loss="sparse_categorical_crossentropy",
metrics=["accuracy"])


# In[35]:


model.fit(train_images, train_labels, epochs=5, batch_size=128)
#model.fit(train_images, train_labels, epochs=100, batch_size=500)


# In[47]:


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


# In[100]:


import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt 
#import ai_functions as func

img00 = np.asarray(Image.open('./mnist/Test_imgs/0_ok.png').convert('L'))
img01 = np.asarray(Image.open('./mnist/Test_imgs/1_ok_1.png').convert('L'))
img02 = np.asarray(Image.open('./mnist/Test_imgs/2_ok_1.png').convert('L'))
img03 = np.asarray(Image.open('./mnist/Test_imgs/3_ok.png').convert('L'))
img04 = np.asarray(Image.open('./mnist/Test_imgs/4_ok.png').convert('L'))

img05 = np.asarray(Image.open('./mnist/Test_imgs/5_ok.png').convert('L'))
img06 = np.asarray(Image.open('./mnist/Test_imgs/6_ok_1.png').convert('L'))
img07 = np.asarray(Image.open('./mnist/Test_imgs/7_ok_1.png').convert('L'))
img08 = np.asarray(Image.open('./mnist/Test_imgs/8_ok_1.png').convert('L'))
img09 = np.asarray(Image.open('./mnist/Test_imgs/9_ok.png').convert('L'))

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

#print('img01.dim =', img01.ndim)
#print('img01.shape =', img01.shape)
#print('img01[0][0] =', img01[0][0])


# In[104]:


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


# In[102]:


predictions = model.predict(img_arr)
#predictions[0]

print('#######################')
for p_num in range (10):
    print('p_num = ', p_num)
    for i in range (10):
          print('is ', i, '---', "%.3f" %(predictions[p_num][i]))  
    print('#######################')

## 7, 8, 9


# In[ ]:




