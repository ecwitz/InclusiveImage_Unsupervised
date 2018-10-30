#from skimage import data, filters
import skimage.data
import numpy as np
import os
import pandas as pd

def import_image(filename):
    #color=skimage.data.load(filename)
    gray=skimage.data.load(filename, as_gray=True)
    print(np.shape(gray))
    return(gray)

def create_matrix():
    cwd = os.getcwd()
    path=cwd+"\\images"
    files = os.listdir(path)
    gray_data=[]
    filenames=[]
    for infile in files:
        filenames.append(infile[:-4])
        gray=import_image(path+"\\"+infile)
        gray_data.append(gray)
    gray_data=np.array(gray_data)
    print(np.shape(gray_data))
    #print(gray_data)
    return(filenames, gray_data)

cwd = os.getcwd()
rawdata=cwd+"\\rawdata"
filenames, data = create_matrix()    
data=create_matrix()   
bdbox=pd.read_csv(rawdata+"\\"+'train-annotations-bboxs.csv')    
print(bdbox.columns.values)  
print(bdbox[[])    
    
'''
REF ONLY
print(filenames)
print(bdbox.columns.values)

print(bdbox[['Source']])
print(bdbox.loc[bdbox['ImageID'] == '4197439585dc5b7bc373o'])



stg1attr=pd.read_csv(rawdata+"\\"+'stage_1_attributions.csv')
print(stg1attr[1:5])
stg1attr.loc[stg1attr['image_id'] == '31677368764467784869413d']

print(stg1attr['image_id'])
print(stg1attr.loc[stg1attr['image_id'].isin(filenames)])
print(bdbox.loc[bdbox['ImageID'].isin(filenames)])
bdbox.shape
''''


