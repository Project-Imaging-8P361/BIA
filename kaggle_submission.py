'''
TU/e BME Project Imaging 2019
Submission code for Kaggle PCAM
Author: Suzanne Wetstein
Last edit: Juul van Boxtel
'''

import os

import numpy as np

import glob
import pandas as pd
from matplotlib.pyplot import imread

from tensorflow.keras.models import model_from_json
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization, WeightNormalization

def kagglesubmission(TEST_PATH, MODEL, MODEL_WEIGHTS, FILEPATH, SAVEPATH):
    # load model and model weights
    MODEL_FILEPATH = os.path.join(FILEPATH, MODEL+'.json')
    MODEL_WEIGHTS_FILEPATH = os.path.join(FILEPATH, MODEL_WEIGHTS+'.hdf5')
   
    json_file = open(MODEL_FILEPATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
        
    # load weights into new model
    model.load_weights(MODEL_WEIGHTS_FILEPATH)
    # open the test set in batches (as it is a very big dataset) and make predictions
    test_files = glob.glob(TEST_PATH + '*.tif')
    
    submission = pd.DataFrame()
    
    file_batch = 500
    max_idx = len(test_files)

    for idx in range(0, max_idx, file_batch):
    
        print('Indexes: %i - %i'%(idx, idx+file_batch))
    
        test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})
    
    
        # get the image id 
        test_df['id'] = test_df.path.map(lambda x: x.split(os.sep)[-1].split('.')[0])
        test_df['image'] = test_df['path'].map(imread)
        
        
        K_test = np.stack(test_df['image'].values)
        
        # apply the same preprocessing as during draining
        K_test = K_test.astype('float')/255.0
        
        predictions = model.predict(K_test)
        
        test_df['label'] = predictions
        submission = pd.concat([submission, test_df[['id', 'label']]])
    
    
    # save your submission
    submission.head()
    submission.to_csv(os.path.join(SAVEPATH, MODEL + ' - submission.csv'), index = False, header = True)

                    

TEST_PATH = r'C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\Q3\OGO imaging\Data\test/'
FILEPATH = r'C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\Q3\OGO imaging group 3\json files'
SAVE_PATH = r'C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\Q3\OGO imaging group 3\json files\submissions' 
MODELS = ['Model_NoNorm_1.', 'Model_batch_1.', 'Model_group_2.', 'Model_Layer_3.','Model_WeightNorm_3.'] 

for MODEL in MODELS:
    for i in range(5):
        MODEL_NAME = MODEL+str(i+1)
        MODEL_WEIGHTS = MODEL_NAME + '_weights'
        print(MODEL_NAME)
        kagglesubmission(TEST_PATH, MODEL_NAME, MODEL_WEIGHTS, FILEPATH, SAVE_PATH)
