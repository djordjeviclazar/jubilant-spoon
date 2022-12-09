import os

running_script_path = os.path.realpath(__file__)
(root_folder, current_file_name) = os.path.split(running_script_path)
data_folder = os.path.join(root_folder, 'data', 'org')
tmp_folder = os.path.join(root_folder, 'tmp')
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

test_folder = os.path.join(data_folder, 'Test')
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
train_folder = os.path.join(data_folder, 'Training')
if not os.path.exists(train_folder):
    os.makedirs(train_folder)

# Every class has different folder with images
classes_list = os.listdir(train_folder) # os.listdir(test_folder)
classes_list.sort()

sourceImages = os.path.realpath(os.path.join(root_folder, '../../../archive/fruits-360_dataset/fruits-360'))
sourceTrainFolder = os.path.join(sourceImages, 'Training')
sourceTestFolder = os.path.join(sourceImages, 'Test')

image_size = 100

batch_size = 16
batch_size_ft = 30
init_lr = 1e-3
init_lr_ft = 1e-5
early_stopping_patience = 30
reduce_lr_patience = 5

model_name = 'D5'#resnet50