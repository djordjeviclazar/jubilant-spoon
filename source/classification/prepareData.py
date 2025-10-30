import os
import shutil

def copyData(sourceDir, destDir, angle=1):

    invalidFiles = 0
    foldersToCopy = ['Training']#, 'Test'
    for folder in foldersToCopy:
        folderPath = os.path.join(sourceDir, folder)

        # create new folder
        destFolderPath = os.path.join(destDir, folder)
        #os.mkdir(destFolderPath)
        # go through all classes
        for classFolder in os.listdir(folderPath):
            classFolderPath = os.path.join(folderPath, classFolder)

            # create class folder
            destClassFolderPath = os.path.join(destFolderPath, classFolder)
            os.mkdir(destClassFolderPath)
            
            # copy files in new folder
            copiedAnglesOfClass = {}
            for file in os.listdir(classFolderPath):
                pictureRotation = ''
                pictureAngle = 0
                fileNameParts = str(file).split('_')
                try:
                    if len(fileNameParts) == 3:
                        pictureRotation = fileNameParts[0]
                        pictureAngle = int(fileNameParts[1])
                    elif len(fileNameParts) == 2:
                        pictureAngle = int(fileNameParts[0])
                    else:
                        scriptName = os.path.realpath('/')
                        raise Exception(f'Custom exception, file {scriptName} line 34. Name of file is "{file}"; 1 or 2 separators: "_" are allowed')
                except Exception as ex:
                    print(f'Custom warning. File name: "{file}" in folder "{classFolderPath}" is not in expected format')
                    invalidFiles += 1

                fileNameHash = pictureRotation + str(int(pictureAngle//angle))
                if (fileNameHash not in copiedAnglesOfClass) or (folder == 'Test'):
                    copiedAnglesOfClass[fileNameHash] = file

                    # copy file
                    filePath = os.path.join(classFolderPath, file)
                    destFilePath = os.path.join(destClassFolderPath, file)
                    shutil.copy(filePath, destFilePath)

            print(classFolder + '\n')
    print('Invalid files: ' + str(invalidFiles))

if True:
    from settings import *
    copyData(sourceDir=sourceImages, destDir=data_folder)

def copy_data_with_augmentation(sourceDir, destDir, angle=5):
    pass


def apply_scale(scale = 1):
    pass

def apply_rotation(angle = 0):
    pass

def apply_translation(percent = 10):
    pass

def apply_dark_box(size_percent = 0.125):
    pass

def apply_salt_and_pepper(ratio = 0.0):
    pass

def apply_gaussian_noise():
    pass

# Test
test_file = 'C:/Users/djord/NotSyncFolder/Repos/Master/DL/archive/fruits-360_dataset/fruits-360/Training/Plum/1_100.jpg'