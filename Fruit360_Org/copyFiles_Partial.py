import os
import shutil
from PIL import Image

sourceDir = 'C:/Users/djord/NotSyncFolder/Repos/Master/DL/archive/fruits-360_dataset/fruits-360'
destDir = 'C:/Users/djord/NotSyncFolder/Repos/Master/DL/jubilant-spoon/Fruit360_Org/Data'

not_RGB_count = 0
foldersToCopy = ['Training', 'Test']
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
        #os.mkdir(destClassFolderPath)
        
        # copy files in new folder
        filesCopied = 0
        for file in os.listdir(classFolderPath):
            #if filesCopied == 50:
            #    break

            filePath = os.path.join(classFolderPath, file)
            img = Image.open(filePath, 'r')
            if img.mode != 'RGB':
                print(filePath)
                not_RGB_count += 1
            #destFilePath = os.path.join(destClassFolderPath, file)
            #shutil.copy(filePath, destFilePath)
            #filesCopied += 1
        print(classFolder + '\n')
print(not_RGB_count)