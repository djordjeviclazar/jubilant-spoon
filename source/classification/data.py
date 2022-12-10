from settings import *
import tensorflow as tf
import numpy as np
import cv2 as cv


class DataProvider(tf.keras.utils.Sequence):

    def __init__(self, images, batch_size):
        self.images = images
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min(len(self.images), (idx + 1) * self.batch_size)
        batch_images = self.images[batch_start:batch_end]

        batch_x = np.zeros((len(batch_images), image_size, image_size, 3), dtype=np.uint8)
        batch_y = ''

        for i in range(len(batch_images)):
            img_path, classIndex = batch_images[i]
            img = getImg_RGB(img_path)
            img = cv.resize(img, (image_size, image_size), interpolation=cv.INTER_LINEAR)
            batch_x[i] = img
            batch_y[i] = classIndex

        return batch_x, batch_y


def getDataProviders(batch_size): # returns tuple (training_data, training_val_data)
    """ Creates and returns data providers for training, validation and test data respectively

    Args:
        batch_size: int

    Returns:
        tuple: (training_data, training_val_data) with type DataProvider
    

    """

    # Build image list
    train_data_provider = {}
    validation_data_provider = {} # validation data for fit function

    # training data
    train_data = []
    validation_data = []
    for classFolder in classes_list:
        testClassPath = os.path.join(train_folder, classFolder)

        cnt = 0
        imagesList = os.listdir(testClassPath)
        for imageName in imagesList:
            img_path = os.path.join(testClassPath, imageName)
            if cnt % 5 == 2:
                validation_data.append((img_path, classes_list.index(classFolder)))
            else:
                train_data.append((img_path, classes_list.index(classFolder)))
            cnt += 1
    
    # Shuffle train data
    train_data = np.random.RandomState(0).permutation(train_data)
    validation_data = np.random.permutation(validation_data)

    # result
    print('Train elements = ' + str(len(train_data)))
    train_data_provider = DataProvider(train_data, batch_size)
    print('Validation elements = ' + str(len(validation_data)))
    validation_data_provider = DataProvider(validation_data, batch_size)

    return (train_data_provider, validation_data_provider)

def getTrainData():
    """ Returns data for training

    Returns:
        tuple: (train_data_x, train_data_y, validation_data_x, validation_data_y)
    
    """

    # training data
    train_data_x = []
    train_data_y = []
    validation_data_x = []
    validation_data_y = []
    for classFolder in classes_list:
        testClassPath = os.path.join(train_folder, classFolder)

        cnt = 0
        imagesList = os.listdir(testClassPath)
        for imageName in imagesList:
            img_path = os.path.join(testClassPath, imageName)
            img = getImg_RGB(img_path)
            #img = cv.resize(img, (image_size, image_size), interpolation=cv.INTER_LINEAR)
            if cnt % 5 == 2:
                validation_data_x.append(img)
                validation_data_y.append(classes_list.index(classFolder))
            else:
                train_data_x.append(img)
                train_data_y.append(classes_list.index(classFolder))
            cnt += 1
    
    # Shuffle train data
    randomState = np.random.RandomState(seed=0).permutation(len(train_data_x))
    train_data_x = (np.asarray(train_data_x))[randomState]
    train_data_y = (np.asarray(train_data_y))[randomState]

    randomState = np.random.RandomState(seed=1024).permutation(len(validation_data_x))
    validation_data_x = (np.asarray(validation_data_x))[randomState]
    validation_data_y = (np.asarray(validation_data_y))[randomState]
    
    return (train_data_x, train_data_y, validation_data_x, validation_data_y)

def getTestData():
    """ Returns test data

    Returns:
        tuple: (test_data_x, test_data_y)
    
    
    """
    # data for evaluation/testing of model
    test_data_x = []
    test_data_y = []
    for classFolder in classes_list:
        testClassPath = os.path.join(train_folder, classFolder)
        imagesList = os.listdir(testClassPath)
        for imageName in imagesList:
            img_path = os.path.join(testClassPath, imageName)
            img = getImg_RGB(img_path)
            #img = cv.resize(img, (image_size, image_size), interpolation=cv.INTER_LINEAR)
            test_data_x.append(img)
            test_data_y.append(classes_list.index(classFolder))
    # result
    
    print('Test elements = ' + str(len(test_data_x)))

    return (np.asarray(test_data_x), np.asarray(test_data_y))

def getImg_RGB(path):
    return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)