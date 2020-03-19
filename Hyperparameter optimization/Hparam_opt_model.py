import os

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD
from tensorboard.plugins.hparams import api as hp
# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32, 64, 128]))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

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
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(hparams[HP_NUM_UNITS], first_kernel, activation='relu', padding='same',input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.MaxPool2D(pool_size=pool_size),
        tf.keras.layers.Conv2D(second_filters, first_kernel, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=pool_size),
        tf.keras.layers.Conv2D(third_filters, second_kernel, activation='relu', padding='valid'),
        tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='valid'),
        tf.keras.layers.Flatten(),
    ])

    # compile the model
    optimizer = SGD(learning_rate=0.01, momentum=0.05, name='SGD')
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def model_training(hparams, epoch, model_name, log_name):
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
    tensorboard = TensorBoard(os.path.join('logs', 'hparam_tuning'), model_name)
    hparamslist = hp.KerasCallback('logs/hparam_tuning', hparams)
    callbacks_list = [checkpoint]

    # train the model
    train_steps = train_gen.n // train_gen.batch_size
    val_steps = val_gen.n // val_gen.batch_size

    history = model.fit(train_gen, steps_per_epoch=train_steps,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  epochs=epoch,
                                  callbacks=callbacks_list)
    print('calculating final accuracy metric')
    _, accuracy = model.evaluate_generator(train_gen, steps=train_steps)
    return accuracy

def run(run_dir, hparams, epoch, model_name, log_name):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = model_training(hparams, epoch, model_name, log_name)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

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
session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams, epoch=1, model_name='Hparamtest1', log_name='hparam')
      session_num += 1