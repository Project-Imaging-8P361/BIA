'''
TU/e BME Project Imaging 2019
Convolutional neural network for PCAM
Author: Suzanne Wetstein
'''

import os

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard

#imports voor mijn twee normalisatietechnieken:
from batch_renorm import BatchRenormalization
from batch_instance_norm import batch_instance_norm

# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

#%%Get data generators
def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

    # dataset parameters
    train_path = os.path.join(base_dir, 'train+val', 'train')
    valid_path = os.path.join(base_dir, 'train+val', 'valid')

    RESCALING_FACTOR = 1./255

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

#%%Get Model
def get_model(model_nr = 1, kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64, conv=False):

    # build the model
    model = Sequential()
    
    if model_nr == 1: #No normalization
        model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(MaxPool2D(pool_size = pool_size))
    
        model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
        model.add(MaxPool2D(pool_size = pool_size))
    
        model.add(Flatten())
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(1, activation = 'sigmoid'))
    
    elif model_nr == 2: #Batch renormalization
        model.add(Conv2D(first_filters, kernel_size, padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(BatchRenormalization())
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size = pool_size))
        
        model.add(Conv2D(second_filters, kernel_size, padding = 'same'))
        model.add(BatchRenormalization())
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size = pool_size))
        
        model.add(Flatten())
                
        model.add(Dense(64, init='uniform'))
        model.add(BatchRenormalization())
        model.add(Activation("relu"))
        
        model.add(Dense(1, init='uniform'))
        model.add(BatchRenormalization())
        model.add(Activation("sigmoid"))
        
#    if model_nr == 3: #Batch Instance normalization
#       model opbouw van het enige voorbeeld is totaal anders 
#       vb: http://easy-tensorflow.com/tf-tutorials/convolutional-neural-nets-cnns?view=category&id=91

    model.compile(SGD(lr=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
    
    return model

#%%
# get the model
def model_training(model, train_gen, val_gen, model_name = 'my_first_cnn_model'):
   
    for layer in model.layers:
        print(layer.output_shape)
    
    # save the model and weights
    model_filepath = model_name + '.json'
    weights_filepath = model_name + '_weights.hdf5'
    
    model_json = model.to_json() # serialize model to JSON
    with open(model_filepath, 'w') as json_file:
        json_file.write(model_json)
    
    # define the model checkpoint and Tensorboard callbacks
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(os.path.join('logs', model_name))
    callbacks_list = [checkpoint, tensorboard]
    
    # train the model
    train_steps = train_gen.n//train_gen.batch_size
    val_steps = val_gen.n//val_gen.batch_size
    
    history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        epochs=1, #tijdelijk naar 1 omdat dit wat sneller is
                        callbacks=callbacks_list)

#%% ROC Analysis
def ROC_analysis(model, test_gen):
    
    test_steps = test_gen.n//test_gen.batch_size
    # ROC analysis
    y_score = model.predict_generator(test_gen, steps=test_steps) #get scores predicted by the model
    y_truth = test_gen.classes #get the ground truths 
    
    fpr, tpr, tresholds = roc_curve(y_truth, y_score) #apply ROC analysis
    roc_auc = auc(fpr, tpr) #calculate area under ROC curve
    
    #create ROC curve plot
    plt.plot(fpr, tpr, [0,1], [0,1], '--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.fill_between(fpr,tpr, alpha = 0.1) #indicate AUC
    plt.text(0.8, 0.03, "AUC = " + str(round(roc_auc, 3)))
    
#%%
filepath = r'C:\Users\20166218\Documents\TU Eindhoven\Jaar 3\Q3\Project imaging\Data'
train_gen, val_gen = get_pcam_generators(filepath)

#model1 = get_model(model_nr=1)
#model_training(model1, train_gen, val_gen, 'Model_1')
#ROC_analysis(model1, val_gen)

model2 = get_model(model_nr=2)
model_training(model2, train_gen, val_gen, 'Model_2')
ROC_analysis(model2, val_gen)

#model3 = get_model(model_nr=3)
#model_training(model3, train_gen, val_gen, 'Model_3')
#ROC_analysis(model3, val_gen)







# tensorboard --logdir logs
# tensorboard --logdir="C:\Users\20166218\Documents\TU Eindhoven\Jaar 3\Q3\Project imaging\Main Project\logs"
# logs elke keer aanpassen en dus ook naar wijzen in logdir

# http://localhost:6006/ 
 




