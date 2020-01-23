import pandas as pd
import json
from utils import *
from train_and_save_model import *
           
def read_data(file):
    
    features = ['TRACK_ID', 'MOOD_TAG', 'AF_PATH']

    train_data = pd.read_csv(file, index_col=False)
    final_data = train_data[features]
    
    return final_data

def dataframe_processing(dataframe, label):
    
    data = dataframe
    classes = {'SAD': 0,'HAPPY': 1} 
    data[label] = [classes[item] for item in data[label] ]
    X = data.drop([label, 'TRACK_ID'], axis = 1)
    y = data[label]

    return data, X, y

def create_training_data(file_path = None):

    # it will hold the training dataframe 
    final = read_data(file_path)
    cols = get_features()
    dfObj = pd.DataFrame(columns=cols)

    for index, row in final.iterrows():    
        
        feature_file_path = row['AF_PATH'] 
        out_dict = populate_features(feature_file_path)
        out_dict['MOOD_TAG'] = row['MOOD_TAG']
        out_dict['TRACK_ID'] = row['TRACK_ID']
        dfObj = dfObj.append(out_dict, ignore_index=True)
        
    label = 'MOOD_TAG'
    file = "temp_train.csv"
    dfObj.to_csv(file, index = None, header=True)
    dfObj = pd.read_csv(file, encoding = "ISO-8859-1")
    
    return dataframe_processing(dfObj, label)
        

if __name__ == "__main__":
    
    import argparse
    import sys
    
    # Initialize parser 
    parser = argparse.ArgumentParser() 
    
    # Adding optional argument 
    parser.add_argument("-file", "--File", help = "Input Training File Path") 
    
    # Read arguments from command line 
    args = parser.parse_args() 
    
    if args.File:  
        data, X, y = create_training_data(file_path = str(args.File))
        model_path = create_and_save_model(X, y)
        print("model Path is : ", model_path)
    else:
        print("Give the training file as input")
        
    
        
        
     
        



    


    
