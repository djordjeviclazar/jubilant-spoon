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

(train_data_x, train_data_y, validation_data_x, validation_data_y) = getTrainData()


########################################################################################################################
# LOADING MODEL
########################################################################################################################

model_version_folder = model_name
model_path = os.path.join(tmp_folder, model_version_folder)
trained_model_path = os.path.join(model_path, 'trained_model.h5')
model = tf.keras.models.load_model(trained_model_path, compile=False,
                                   custom_objects={'tf': tf})

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=init_lr_ft),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

model.summary()


########################################################################################################################
# TRAINING MODEL
########################################################################################################################

path = os.path.join(model_path, 'fine_tuned_model.h5')
save_model = tf.keras.callbacks.ModelCheckpoint(path, monitor='val_sparse_categorical_accuracy', mode='max',
                                                verbose=1, save_best_only=True)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max',
                                                  patience=10, restore_best_weights=True, verbose=1)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', mode='max',
                                                 factor=0.1, patience=5, verbose=1)

csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_path, 'fine_tuning.csv'))

hist = model.fit(train_data_x, train_data_y, batch_size=batch_size_ft,
                 epochs=50,
                 validation_data=(validation_data_x, validation_data_y),
                 callbacks=[save_model, early_stopping, reduce_lr, csv_logger],
                 verbose=1)

model.save(path, include_optimizer=False)

plt.clf()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.savefig(os.path.join(model_path, 'fine_tuning_loss.png'))

plt.clf()
plt.plot(hist.history['sparse_categorical_accuracy'])
plt.plot(hist.history['val_sparse_categorical_accuracy'])
plt.savefig(os.path.join(model_path, 'fine_tuning_accuracy.png'))