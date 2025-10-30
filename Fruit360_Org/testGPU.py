# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

#import tensorflow as tf 
#print(tf.__version__)
#
#print('1: ', tf.config.list_physical_devices('GPU'))
#print('2: ', tf.test.is_built_with_cuda)
#print('3: ', tf.test.gpu_device_name())
#print('4: ', tf.config.get_visible_devices())

#import tensorflow as tf 
#print(tf.config.list_physical_devices('GPU'))

#class Test:
#    def __init__(self):
#        testVar = 'aaaaaa'
#        self.testPrivateVar = 'asdasd'
#
#    def Print(self):
#        print(self.testPrivateVar)
#        #print(self.testVar)
#
#a = Test()
#a.Print()

#import os
#print(os.path.realpath('./'))
#print(os.path.realpath(os.path.join(os.path.realpath('./'), '../../')))

#import numpy as np
#validation_data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#print (validation_data)
#validation_data = np.random.permutation(validation_data)
#print (validation_data)

#from PIL import Image
#filePath = 'C:/Users/djord/NotSyncFolder/Repos/Master/DL/archive/fruits-360_dataset/fruits-360/Training/Plum/1_100.jpg'
#img = Image.open(filePath, 'r')
#print(img.mode)

import cv2 as cv
#print(cv.imread(filePath))

import numpy as np
#testList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#testList2 = [11, 21, 31, 41, 51, 61, 71, 81, 91]
#randomState = np.random.RandomState(seed=1024)
#result1 = randomState.permutation(testList)
#result2 = randomState.permutation(testList2)

#randomState = np.random.RandomState(seed=1024).permutation(len(testList))
#result1 = testList[randomState]
#result2 = testList2[randomState]

#print(result1)
#print(result2)

test_data_x = []
test_data_x.append(cv.imread('C:/Users/djord/NotSyncFolder/Repos/Master/DL/archive/fruits-360_dataset/fruits-360/Training/Plum/1_100.jpg'))
test_np = np.asarray(test_data_x)

fileOrg = open('C:/Users/djord/NotSyncFolder/Repos/Master/DL/test_org.txt', 'w')
fileNp = open('C:/Users/djord/NotSyncFolder/Repos/Master/DL/test_np.txt', 'w')

#fileOrg.write(str(test_data_x))
for line in test_data_x:
    fileOrg.write(f"{line}\n")
np.savetxt(fileNp, )
#fileNp.write(str(np.asarray(test_data_x)))