
# coding: utf-8

# In[2]:

# The dataset that we are using for the project consists of 289205 .wav files belonging to the Training Audio Set and 4096 .wav
# files belonging to the Testing Audio Set. The purpose of collecting audio files is to generate new audio by fitting on the
# training data set and then predicting using the Testing data Set using GAN.


# In[4]:

import scipy.constants as const
import scipy
from scipy.io import wavfile
from IPython.core.display import HTML
from __future__ import division
import glob,os
import pip
import librosa
get_ipython().magic('matplotlib inline')
import seaborn # optional
import matplotlib.pyplot as plt
import librosa.display
import struct
import pandas as pd
import csv
import wave
import json
get_ipython().magic('pylab inline')


# In[3]:

import pip
package_name='boto'
pip.main(['install', package_name])


# In[ ]:

# Since the data is in .wav format, we write a code to unpack some features of the audio file.


# # Code to convert all the audio files to numeric data which gives the details of the audio files such as sample rate, frequency, byte rate, instrument source, instrument type etc.

# In[3]:

directoryPath = 'C://Users//chels//Desktop//nsynth-train.jsonwav//'
with open('csvTrainAssign3.csv', "w",encoding='utf-8',newline='') as output:
    writer = csv.writer(output)
    writer.writerow (['filename','ChunkID','TotalSize','DataSize','Format','SubChunk1ID','SubChunk1Size','AudioFormat','NumChannels','SampleRate','ByteRate','BlockAlign','BitsPerSample','SubChunk2ID','SubChunk2Size','S1','S2','S3','S4','S5'])
    for dire,subdir,files in os.walk(directoryPath):
        for filename in files:
            content = open(os.path.join(dire, filename), 'rb')
            #print(content)
            ChunkID=content.read(4) # First four bytes are ChunkID which must be "RIFF" in ASCII
            #print("ChunkID=",ChunkID)
            ChunkSizeString=content.read(4) # Total Size of File in Bytes - 8 Bytes
            ChunkSize=struct.unpack('I',ChunkSizeString) # 'I' Format is to to treat the 4 bytes as unsigned 32-bit inter
            TotalSize=ChunkSize[0]+8 # The subscript is used because struct unpack returns everything as tuple
            #print("TotalSize=",TotalSize)
            DataSize=TotalSize-44 # This is the number of bytes of data
            #print("DataSize=",DataSize)
            Format=content.read(4) # "WAVE" in ASCII
            #print("Format=",Format) 
            SubChunk1ID=content.read(4) # "fmt " in ASCII
            #print("SubChunk1ID=",SubChunk1ID)
            SubChunk1SizeString=content.read(4) # Should be 16 (PCM, Pulse Code Modulation)
            SubChunk1Size=struct.unpack("I",SubChunk1SizeString) # 'I' format to treat as unsigned 32-bit integer
            #print("SubChunk1Size=",SubChunk1Size[0])
            AudioFormatString=content.read(2) # Should be 1 (PCM)
            AudioFormat=struct.unpack("H",AudioFormatString) # 'H' format to treat as unsigned 16-bit integer
            #print("AudioFormat=",AudioFormat[0])
            NumChannelsString=content.read(2) # Should be 1 for mono, 2 for stereo
            NumChannels=struct.unpack("H",NumChannelsString) # 'H' unsigned 16-bit integer
            #print("NumChannels=",NumChannels[0])
            SampleRateString=content.read(4) # Should be 44100 (CD sampling rate)
            SampleRate=struct.unpack("I",SampleRateString)
            #print("SampleRate=",SampleRate[0])
            ByteRateString=content.read(4) # 44100*NumChan*2 (88200 - Mono, 176400 - Stereo)
            ByteRate=struct.unpack("I",ByteRateString) # 'I' unsigned 32 bit integer
            #print("ByteRate=",ByteRate[0])
            BlockAlignString=content.read(2) # NumChan*2 (2 - Mono, 4 - Stereo)
            BlockAlign=struct.unpack("H",BlockAlignString) # 'H' unsigned 16-bit integer
            #print("BlockAlign=",BlockAlign[0])
            BitsPerSampleString=content.read(2) # 16 (CD has 16-bits per sample for each channel)
            BitsPerSample=struct.unpack("H",BitsPerSampleString) # 'H' unsigned 16-bit integer
            #print("BitsPerSample=",BitsPerSample[0])
            SubChunk2ID=content.read(4) # "data" in ASCII
            #print("SubChunk2ID=",SubChunk2ID)
            SubChunk2SizeString=content.read(4) # Number of Data Bytes, Same as DataSize
            SubChunk2Size=struct.unpack("I",SubChunk2SizeString)
            #print("SubChunk2Size=",SubChunk2Size[0])
            S1String=content.read(2) # Read first data, number between -32768 and 32767
            S1=struct.unpack("h",S1String)
            #print("S1=",S1[0])
            S2String=content.read(2) # Read second data, number between -32768 and 32767
            S2=struct.unpack("h",S2String)
            #print("S2=",S2[0])
            S3String=content.read(2) # Read second data, number between -32768 and 32767
            S3=struct.unpack("h",S3String)
            #print("S3=",S3[0])
            S4String=content.read(2) # Read second data, number between -32768 and 32767
            S4=struct.unpack("h",S4String)
            #print("S4=",S4[0])
            S5String=content.read(2) # Read second data, number between -32768 and 32767
            S5=struct.unpack("h",S5String)
            #print("S5=",S5[0])
            content.name
            
            content.close()
            
            res = [filename,ChunkID,TotalSize,DataSize,Format,SubChunk1ID,SubChunk1Size[0],AudioFormat[0],NumChannels[0],
                        SampleRate[0],ByteRate[0],BlockAlign[0],BitsPerSample[0],SubChunk2ID,SubChunk2Size[0],S1[0],S2[0],S3[0],S4[0],S5[0]]
            writer.writerow(res)            
           


# # Code for writing json to csv

# In[2]:

a=pd.read_json('examples_test.json', convert_axes= False)
data=a.T
data.head()
#COnverting Test JSON file to CSV


# In[ ]:

# Here why there are two CSV conversions are because in one csv file 'Test_Data_csv.csv', there is unpacked data which was 
# extracted by applying .unpack function on all the audio files of the TEST folder. The other json file called 
# 'examples_test.json' has the above information has also been converted to 'testjsontocsv.csv'.  


# # Code for merging test data CSV's

# In[43]:

test_csv_freq=pd.read_csv('Test_Data_csv.csv')
test_csv_freq
test_csv_json=pd.read_csv('testjsontocsv.csv')
test_csv_json

test_csv_json['filename'] =  test_csv_json['filename']+ '.wav'
test_csv_json
#In this step, we are adding .wav at the end of every filename in order to maintain uniformity amidst both the gathered sources.


# In[ ]:

# The reason we are merging both the CSV files is to gather all the information about the audio files in the TEST folder in one 
# csv. The 'merged_test.csv' has 33 columns worth information from the two single CSV's which were obtained by applying the
# .unpack function and the JSON file.


# In[196]:

result = pd.concat([test_csv_json,test_csv_freq], axis=1, join_axes=[test_csv_freq.filename])
result

merged_test=test_csv_json.merge(test_csv_freq, on='filename', how='inner')
merged_test.to_csv('merged_test.csv')
#This is the CSV which has all the required information about the audio files in the TEST folder in numeric and string format.


# # Code for merging train data 

# In[47]:

train_csv_freq=pd.read_csv('Train_Data_csv.csv')
train_csv_freq
train_csv_json=pd.read_csv('trainjsontocsv.csv')
train_csv_json

train_csv_json['filename'] =  train_csv_json['filename']+ '.wav'
train_csv_json
#Similarly, we do the same for gathering all the data for the audio files in the TRAIN folder.
#In this snippet, we are adding .wav at the end of the filename to maintain uniformity in all the CSV files.


# In[195]:

result1 = pd.concat([train_csv_json,train_csv_freq], axis=1, join_axes=[train_csv_freq.filename])
result1

merged_train=train_csv_json.merge(train_csv_freq, on='filename', how='inner')
merged_train.to_csv('merged_train.csv')
#This file has all the required information which has been gathered from the code which used .unpack function and the JSON file.
# This file has also 33 columsn worth data which are numeric and of string type.


# # Code for converting csv to JSON (train)

# In[ ]:

# Since, there are already exisiting two JSON files for both the TEST and the TRAIN folder, we decided to convert the merged CSV's
# of Test and Train into JSON files which would further be split into multiplt JSON files thus acting as a single record 
# in a collection which will be made on MongoDB.


# In[156]:

csvfile = open('merged_train.csv', 'r')
jsonfile = open('train_data_json.json', 'w')

fieldnames = ("filename","instrument","instrument_family","instrument_family_str","instrument_source","instrument_source_str",
              "instrument_str","note","note_str","pitch","qualities","qualities_str","sample_rate","velocity",
              "ChunkID","TotalSize","DataSize","Format","SubChunk1ID","SubChunk1Size","AudioFormat","NumChannels",
              "SampleRate","ByteRate","BlockAlign","BitsPerSample","SubChunk2ID","SubChunk2Size","S1","S2","S3","S4","S5")

reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')


# # Code for converting csv to JSON (test)

# In[151]:

csvfile = open('merged_test1.csv', 'r')
jsonfile = open('test_data_json.json', 'w')

fieldnames = ("filename","instrument","instrument_family","instrument_family_str","instrument_source","instrument_source_str",
              "instrument_str","note","note_str","pitch","qualities","qualities_str","sample_rate","velocity",
              "ChunkID","TotalSize","DataSize","Format","SubChunk1ID","SubChunk1Size","AudioFormat","NumChannels",
              "SampleRate","ByteRate","BlockAlign","BitsPerSample","SubChunk2ID","SubChunk2Size","S1","S2","S3","S4","S5")

reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')


# # splitting json to multiple jsons (train)

# In[ ]:

# Here we are splitting the main train JSON file into multiple JSON files based on the number of entries in the main train JSON
# file The count is 289205 and there are 289205 single JSON files being split from the main one.


# In[159]:

counter=1
path='C:\\Users\\chels\\Assign3\\Assignment3_data\\train_data\\trainjson'
with open ('train_data_json.json', 'r') as f:
    for line in f:
        a=json.loads(line)
        newpath=os.path.join(path,str(counter)+'.json')
        with open(newpath,'w') as json_file:
            json.dump(a,json_file)
            counter+=1


# # splitting json to multiple jsons (test)

# In[ ]:

# Here we are splitting the main test JSON file into multiple JSON files based on the number of entries in the main test JSON
# file The count is 4096 and there are 4096 single JSON files being split from the main one.


# In[ ]:

counter=1
path='C:\\Users\\chels\\Assign3\\Assignment3_data\\test_data\\json_test'
with open ('test_data_json.json', 'r') as f:
    for line in f:
        a=json.loads(line)
        newpath=os.path.join(path,str(counter)+'.json')
        with open(newpath,'w') as json_file:
            json.dump(a,json_file)
            counter+=1


# # Created the required files for exporting them to mongoDB to make it into a database. The library being used here is pymongo

# In[ ]:

#Since MongoDB is a NoSql database, most of the data is of the form of a dictionary with every record having one key and a value.
#This was the reason to merge different CSV's to one and then convert into JSON and split it. The entire process was to gather
# enough data to load into MongoDB.


# # Establishing connection to MongoDB hosted on an Atlas Cluster. The instance size is M0 and has a 3 node Architecture.
# 

# In[ ]:

# The link to the cloud server is : https://cloud.mongodb.com/v2/59f8ec6fd383ad3f3ee771f0#clusters?tooltip=nds.connect&step=0


# In[19]:

from pymongo import MongoClient
# pprint library is used to make the output look more pretty
from pprint import pprint
# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient("mongodb://pranavcfc:7AwzYVpIO4BP7hKp@cluster0-shard-00-00-vkqzr.mongodb.net:27017,cluster0-shard-00-01-vkqzr.mongodb.net:27017,cluster0-shard-00-02-vkqzr.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin")
print (client.mflix)
db=client.admin
# Issue the serverStatus command and print the results
serverStatusResult=db.command("serverStatus")
pprint(serverStatusResult)


# In[20]:

print (client)


# In[21]:

client = MongoClient('localhost', 27017)
print(client)


# In[22]:

db = client.test_database
print(db)


# # Part A - Create a database

# # Insertion of json to MongoDB (Test)

# In[ ]:

# Here we are inserting the 4096 split JSON files into MongoDB and storing it in 'audio_test_collection' and invoking that
# db through a variable called record1.


# In[23]:

import json
import pymongo
connection = pymongo.MongoClient("mongodb://pranavcfc:7AwzYVpIO4BP7hKp@cluster0-shard-00-00-vkqzr.mongodb.net:27017,cluster0-shard-00-01-vkqzr.mongodb.net:27017,cluster0-shard-00-02-vkqzr.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin")
db=connection.audio_train
record1 = db.audio_test_collection


# In[11]:


directoryPath="C://Users//chels//Assign3//Assignment3_data//test_data//json_test"
for dire,subdir,files in os.walk(directoryPath):
    for filename in files:
        page = open(os.path.join(dire, filename), 'r')
        parsed = json.loads(page.read())
        record1.insert(parsed)


# In[77]:

collection = db['audio_test_collection']


# In[78]:

print(collection)


# In[24]:

import pprint
pprint.pprint(record1.find_one())


# In[25]:

pprint.pprint(record1.count())
# The 4096 number has been varied because there were insertions twice and other CRUD operations were performed on the 
# database which has faltered the number.


# # insertion of json to mongo db (train)

# In[ ]:

# Here we are inserting the 289205 split JSON files into MongoDB and storing it in 'audio_train_collection' and invoking that
# db through a variable called record2.


# In[ ]:


directoryPath="C://Users//chels//Assign3//Assignment3_data//train_data//trainjson"
for dire,subdir,files in os.walk(directoryPath):
    for filename in files:
        page = open(os.path.join(dire, filename), 'r')
        parsed = json.loads(page.read())
        record2.insert(parsed)


# In[81]:

connection = pymongo.MongoClient("mongodb://pranavcfc:7AwzYVpIO4BP7hKp@cluster0-shard-00-00-vkqzr.mongodb.net:27017,cluster0-shard-00-01-vkqzr.mongodb.net:27017,cluster0-shard-00-02-vkqzr.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin")
db=connection.audio_train
#record1 = db.audio_test_collection
record2 = db.audio_train_collection


# In[82]:

collection = db['audio_train_collection']


# In[83]:

pprint.pprint(record2.count())
# # The 289205 number has been varied because there were multiple insertions and other CRUD operations were performed on the 
# database which has faltered the number.


# # Part B - Queries

# # Query 1 - Selecting query to return  one BSON document.

# In[84]:

pprint.pprint(record2.find_one())
# This is a simple select query where it just returns one query from the Test collection.


# In[85]:

pprint.pprint(record2.find_one({"filename" : "ok"}))
# It returns none when it does not find the filename.


# In[86]:

pprint.pprint(record2.find_one({"filename" : "bass_synthetic_004-065-100.wav"}))
#This is a specific search where the result retrieved is by the filename.


# In[96]:

for post in record1.find({'ByteRate': '32000'}):
    pprint.pprint(post)
# The key difference between using find and find_one is that find results is multiple searches and find_one gives us one 
# particular search out of the many searches that it has retrieved.


# # Query 2 - Create a row and add it to the test collection.

# In[88]:

post= {'AudioFormat': '2',
 'BitsPerSample': '16',
 'BlockAlign': '2',
 'ByteRate': '32000',
 'ChunkID': "b'RIFF'",
 'DataSize': '128000',
 'Format': "b'WAVE'",
 'NumChannels': '1',
 'S1': '0',
 'S2': '0',
 'S3': '0',
 'S4': '0',
 'S5': '0',
 'SampleRate': '16000',
 'SubChunk1ID': "b'fmt '",
 'SubChunk1Size': '16',
 'SubChunk2ID': "b'data'",
 'SubChunk2Size': '128000',
 'TotalSize': '128044',
 'filename': 'bass_synthetic_new.wav',
 'instrument': '45',
 'instrument_family': '0',
 'instrument_family_str': 'bass',
 'instrument_source': '2',
 'instrument_source_str': 'synthetic',
 'instrument_str': 'bass_synthetic_004',
 'note': '213052',
 'note_str': 'bass_synthetic_004-065-100',
 'pitch': '65',
 'qualities': '[0, 0, 1, 0, 0, 1, 0, 0, 0, 0]',
 'qualities_str': "['distortion', 'multiphonic']",
 'sample_rate': '16000',
 'velocity': '100'}

post_id = record2.insert_one(post).inserted_id
post_id
#Here we are inserting a new record which can be considered as good as a splt JSON file which is newly being added.


# In[89]:

for post in record2.find({'AudioFormat': '2'}):
    pprint.pprint(post)
    
#Here in order to check whether the new file has been entered, we query to cross-check.


# # Query 3 - Update a record in test or train collection

# In[90]:

record1.update_one(
    {'filename': 'bass_electronic_018-022-100.wav'},
    {
        "$set": {'instrument': '999'}
    })
# Here we updated a BSON document of record1 collection and set the instrument as 799 of the
# filename :  'bass_electronic_018-022-100.wav'


# In[91]:

record1.find_one({'filename': 'bass_electronic_018-022-100.wav'})
# In Order to check whether the BSON document was updated, we query to see.


# # Query 4 -Delete a record from test or train collection

# In[92]:

# delete the document where 'note': '213052'
result = record2.delete_one({'instrument': '799'})
print(result)


# # Query 5 - Sorting A Document

# In[93]:

result = record1.find().sort([("instrument", pymongo.DESCENDING)])
result.next()
# This query sorts the documents by 'instrument' and retrieves the last audio file BSON in our collection.


# # execution time of a query

# In[95]:

#db.restaurants.explain("executionStats") (finding a document)
record2.find({'note': '213052'}).explain()['executionStats']
# Here the time executed can be seen by the 'executionTimeMillisEstimate': 181 which is 181 milliseconds.


# In[ ]:

# The execution time of the other queries will be shown in Atlas MongoDB Compass


# In[ ]:

# Uploading data to Amazon S3


# In[ ]:


 


# In[ ]:


   

