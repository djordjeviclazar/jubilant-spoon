from data import *
import matplotlib.pyplot as plt


print("Tensorflow version: " + tf.__version__)
print("Keras version: " + tf.keras.__version__)

tf.compat.v1.disable_eager_execution()

# FORCE CPU EXECUTION
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


########################################################################################################################
# LOADING DATA
########################################################################################################################

(test_data_x, test_data_y) = getTestData()


########################################################################################################################
# LOADING MODEL
########################################################################################################################

model_version_folder = model_name
model_path = os.path.join(tmp_folder, model_version_folder)
fine_tuned_model_path = 'c:/Users/djord/NotSyncFolder/Repos/Master/DL/jubilant-spoon/Fruit360_Org/output_files/fruit-360 model_selected_data/model.h5'#os.path.join(model_path, 'trained_model.h5')#fine_tuned_model#trained_model

model = tf.keras.models.load_model(fine_tuned_model_path, compile=False,
                                   custom_objects={'tf': tf})

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=init_lr),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

model.summary()


########################################################################################################################
# EVALUATE MODEL
########################################################################################################################

#res_train = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
res_test = model.evaluate(test_data_x, test_data_y, batch_size=batch_size_eval, verbose=0)
#print(res_train)
print(model.metrics_names)
print(res_test)
with open(os.path.join(model_path, 'evaluate.txt'), mode='at') as f:
    print(model.metrics_names, file=f)
    print(res_test, file=f)

print('test')

#y_out = model.predict(test_data_x, batch_size=batch_size)
#y_out = np.argmax(y_out, axis=1)
#i = 0
#plt.figure(figsize=(4, 4))
#for img, out, exp in zip(test_data_x, y_out, test_data_y):
#    if out != exp:
#        plt.clf()
#        plt.imshow(img)
#        title = '{} as {}'.format(classes_list[int(exp)], classes_list[int(out)])
#        plt.title(title)
#        i += 1
#        plt.savefig(os.path.join(model_path, '{} ({}).jpg'.format(i, title)))