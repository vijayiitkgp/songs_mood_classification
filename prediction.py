import pandas as pd
import json
from train_and_save_model import *
from utils import *
from sklearn.externals import joblib

def read_data(file):
    
    features = ['TRACK_ID', 'AF_PATH']
    test_data = pd.read_csv(file, index_col=False)
    final_data = test_data[features]

    return test_data, final_data

def dataframe_processing(dataframe):
    
    data = dataframe
    X = data.drop(['TRACK_ID'], axis = 1)

    return X

def create_test_data(file_path = None):

    # it will hold the test dataframe 
    test_data, final = read_data(file_path)
    cols = get_features()
    dfObj = pd.DataFrame(columns=cols) 

    for index, row in final.iterrows():    

        feature_file_path = row['AF_PATH'] 
        out_dict = populate_features(feature_file_path)
        out_dict['TRACK_ID'] = row['TRACK_ID']
        dfObj = dfObj.append(out_dict, ignore_index=True)
    
    file = "temp_test.csv"
    dfObj.to_csv(file, index = None, header=True)
    dfObj = pd.read_csv(file, encoding = "ISO-8859-1")
    
    return test_data, dataframe_processing(dfObj)

if __name__ == "__main__":
    
    import argparse
    import sys
    
    # Initialize parser 
    parser = argparse.ArgumentParser() 
    
    # Adding optional argument 
    parser.add_argument("-file", "--File", help = "Input Test File Path") 
    parser.add_argument("-model", "--Model", help = "Input Model Path")
    
    # Read arguments from command line 
    args = parser.parse_args() 
  
    if args.File:
        final_test, X= create_test_data(file_path = str(args.File))
        
    if args.Model:  
        model = joblib.load(str(args.Model))  
        y_pred_test = model.predict(X)
        classes = {0:'SAD', 1: 'HAPPY'} 
        final_test['MOOD_TAG'] = y_pred_test
        final_test['MOOD_TAG'] = [classes[item] for item in final_test['MOOD_TAG'] ]
        final_test.to_csv(str(args.File), index = None, header=True)

    
        
    
        
        
     
        



    


    

    




    


    
