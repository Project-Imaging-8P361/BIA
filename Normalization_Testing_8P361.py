'''
TU/e BME Project Imaging 2019
Convolutional neural network for PCAM
Group 3

'''
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Flatten, Conv2D, MaxPool2D, BatchNormalization, LayerNormalization
from tensorflow_addons.layers import GroupNormalization, WeightNormalization

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
def get_model(model_nr = 1, norm_layer = "", option = 0, kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):
    
    if model_nr != 1 and norm_layer == "":
        print("Please define a normalization layer or choose model_nr= 2 or 3")
        return
    if model_nr != 1 and option == 0:
        print("Please define the position of the normalization layer in argument 'option'.")
        return
    if model_nr == 3 and norm_layer == WeigthNormalization:
        print("For Model 3 please choose WeightNormalization as normalization layer, or choose other model nr.")
        return
       
    # build the model
    model = Sequential()
    
    if model_nr == 1: #model without normalization
        #input layer
        model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(MaxPool2D(pool_size = pool_size))
        
        #hidden layer 1
        model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
        model.add(MaxPool2D(pool_size = pool_size))
        
        #hidden layer 2
        model.add(Conv2D(64, (6, 6), activation = 'relu', padding = 'valid'))   # Convolutional layer acting as fully connected layer (kernel size == input size)
        
        #output layer
        model.add(Conv2D(1, (1, 1), activation='sigmoid', padding='valid'))     # Convolutional layer acting as fully connected output layer (kernel size == input size)
        model.add(Flatten())         
    elif model_nr == 2: #Batch, Group, Layer Normalization
        #input layer
        model.add(Conv2D(first_filters, kernel_size, padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
        if option == 1 or option == 3: model.add(norm_layer)
        model.add(Activation('relu'))
        if option == 4 or option == 6: model.add(norm_layer)
        model.add(MaxPool2D(pool_size = pool_size))
        
        #hidden layer 1
        model.add(Conv2D(second_filters, kernel_size, padding = 'same'))
        if option == 2 or option == 3: model.add(norm_layer)
        model.add(Activation('relu'))
        if option == 5 or option == 6: model.add(norm_layer)
        model.add(MaxPool2D(pool_size = pool_size))
        
        #hidden layer 2
        model.add(Conv2D(64, (6, 6), padding = 'valid'))   # Convolutional layer acting as fully connected layer (kernel size == input size)
        if option == 2 or option == 3: model.add(norm_layer)
        model.add(Activation('relu'))
        if option == 5 or option == 6: model.add(norm_layer)
        
        #output layer
        model.add(Conv2D(1, (1, 1), activation='sigmoid', padding='valid'))     # Convolutional layer acting as fully connected output layer (kernel size == input size)
        model.add(Flatten())         
    elif model_nr == 3: #Weight normalization
        #input layer
        if option == 1 or option == 3: model.add(norm_layer(Conv2D(first_filters, kernel_size, padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3))))
        elif option == 2: model.add(Conv2D(first_filters, kernel_size, padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size = pool_size))
        
        #hidden layer 1
        if option == 2 or option ==3: model.add(norm_layer(Conv2D(second_filters, kernel_size, padding = 'same')))
        elif option == 1: model.add(Conv2D(second_filters, kernel_size, padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size = pool_size))
        
        #hidden layer 2
        if option == 2 or option ==3: model.add(norm_layer(Conv2D(64, (6, 6), padding = 'valid')))
        elif option == 1: model.add(Conv2D(64, (6, 6), padding = 'valid'))
        model.add(Activation('relu'))
        
        #output layer
        model.add(Conv2D(1, (1, 1), activation='sigmoid', padding='valid'))     # Convolutional layer acting as fully connected output layer (kernel size == input size)
        model.add(Flatten())
    else:
        print("Invalid model nr. Please try again.")
        return
    
     # compile the model
    model.compile(SGD(lr=0.01, momentum=0.05), loss = 'binary_crossentropy', metrics=['accuracy'])
    
    return model


#%%
# get the model
def model_training(model, train_gen, val_gen, model_name = 'my_first_cnn_model', verbose = 1):
   
    #for layer in model.layers:
    #    print(layer.output_shape)
    
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
                        epochs=15,
                        callbacks=callbacks_list, 
                        verbose = verbose)
    return history
    

#%% ROC Analysis
def ROC_analysis(model, test_gen):
    
    test_steps = test_gen.n//test_gen.batch_size
    # ROC analysis
    y_score = model.predict(test_gen, steps=test_steps) #get scores predicted by the model
    y_truth = test_gen.classes #get the ground truths 
    
    fpr, tpr, tresholds = roc_curve(y_truth, y_score) #apply ROC analysis
    roc_auc = auc(fpr, tpr) #calculate area under ROC curve
    
    #create ROC curve plot
    plt.figure()
    plt.plot(fpr, tpr, [0,1], [0,1], '--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.fill_between(fpr,tpr, alpha = 0.1) #indicate AUC
    plt.text(0.8, 0.03, "AUC = " + str(round(roc_auc, 3)))
    return roc_auc

#%%
filepath = r'C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\Q3\OGO imaging\Data'

'''
Options:
model_nr = 1 --> basic model without normalization
model_nr = 2 --> model with normalization layer (BatchNorm, LayerNorm, GroupNorm)
model_nr = 3 --> model with weight normalization

norm_layer = BatchNormalization()
norm_layer = LayerNormalization()
norm_layer = GroupNormalization()
norm_layer = WeightNormalization

option = 1 --> Normalization before activation on input layer
option = 2 --> Normalization before activation on hidden layers
option = 3 --> Normalization before activation on input layer and hidden layers
option = 4 --> Normalization after activation on input layer
option = 5 --> Normalization after activation on hidden layers
option = 6 --> Normalization after activation on input layer and hidden layers
'''
#model_basic = get_model(model_nr=1)
#model_instance = get_model(model_nr=2, norm_layer = InstanceNormalization(axis=3,center=True,scale=True,beta_initializer="random_uniform",gamma_initializer="random_uniform"))
#model_batch = get_model(model_nr=2, norm_layer = BatchNormalization())
#model_layer = get_model(model_nr=2, norm_layer = LayerNormalization())
#model_group = get_model(model_nr=2, norm_layer = GroupNormalization(groups=32, axis=3))
#model_weight = get_model(model_nr=3, norm_layer = WeightNormalization)

train_gen, val_gen = get_pcam_generators(filepath)
model = get_model(model_nr = 2, norm_layer = BatchNormalization(), option = 1)
history = model_training(model, train_gen, val_gen, 'Model_BatchNorm_1.1', verbose=1)
auc_score = ROC_analysis(model, val_gen)

