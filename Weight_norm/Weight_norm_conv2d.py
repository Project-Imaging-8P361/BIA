import os

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Lambda
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD
from Weight_norm.conv2dweight import conv2d_weight_norm as Conv2DWeightNorm

# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):
    # dataset parameters
    train_path = os.path.join(base_dir, 'train')
    valid_path = os.path.join(base_dir, 'valid')

    RESCALING_FACTOR = 1. / 255

    # instantiate data generators
    datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

    train_gen = datagen.flow_from_directory(train_path,
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size=train_batch_size,
                                            class_mode='binary')

    val_gen = datagen.flow_from_directory(valid_path,
                                          target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                          batch_size=val_batch_size,
                                          class_mode='binary', shuffle=False)

    return train_gen, val_gen


def get_model(first_kernel=(3, 3), second_kernel = (6,6), pool_size=(4, 4), first_filters=32, second_filters=64, third_filters=64):
    # build the model
    def conv2dweight1(x):
        return Conv2DWeightNorm(x, filters=first_filters, kernel_size=first_kernel, activation='relu', padding='same')[0]

    def conv2dweight2(x):
        return Conv2DWeightNorm(x, filters=second_filters, kernel_size=first_kernel, activation='relu', padding='same')

    def conv2dweight3(x):
        return Conv2DWeightNorm(x, filters=third_filters, kernel_size=second_kernel, activation='relu', padding='valid')


    model = Sequential()

    model.add(Conv2D(filters=first_filters, kernel_size=first_kernel, activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Lambda(conv2dweight2))

    model.add(Lambda(conv2dweight3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    # compile the model
    model.compile(SGD(lr=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

    return model


def model_training(epoch, model_name, log_name):
    # get model
    model = get_model()

    # get data
    filepath = r'C:/Users/20172960/Documents/Project imaging Data/Data'
    train_gen, val_gen = get_pcam_generators(filepath)

    # save the model and weights
    model_filepath = model_name + '.json'
    weights_filepath = model_name + '_weights.hdf5'

    model_json = model.to_json()  # serialize model to JSON
    with open(model_filepath, 'w') as json_file:
        json_file.write(model_json)

    # define the model checkpoint and Tensorboard callbacks
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(os.path.join(log_name, model_name))
    callbacks_list = [checkpoint, tensorboard]

    # train the model
    train_steps = train_gen.n // train_gen.batch_size
    val_steps = val_gen.n // val_gen.batch_size

    history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  epochs=epoch,
                                  callbacks=callbacks_list)


def ROC_analysis(model, test_gen):
    test_steps = test_gen.n // test_gen.batch_size
    # ROC analysis
    y_score = model.predict_generator(test_gen, steps=test_steps)  # get scores predicted by the model
    y_truth = test_gen.classes  # get the ground truths

    fpr, tpr, tresholds = roc_curve(y_truth, y_score)  # apply ROC analysis
    roc_auc = auc(fpr, tpr)  # calculate area under ROC curve

    # create ROC curve plot
    plt.plot(fpr, tpr, [0, 1], [0, 1], '--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.fill_between(fpr, tpr, alpha=0.1)  # indicate AUC
    plt.text(0.8, 0.03, "AUC = " + str(round(roc_auc, 3)))


##
model_training(1,'Weightnorm2','logWL')