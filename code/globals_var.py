


#=====================================  Libraries  ======================================#
#========================================================================================#

import os
import shutil
import glob
import click 
import ast
import remo

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models

#========================================================================================#
#========================================================================================#




#=====================================  Globals  ========================================#
#========================================================================================#

## 1. Not to change.
base_path = os.getcwd()[:-5]
all_annotation = 'Image classification'

## 2. Change if ou want.
dataset_name = 'project'
predicaion_annotation = 'model prediction'
model_name = 'AlexNetmodel.pb'
Classes = { 0 : 'Cat',  
            1 : 'Dog' } 

# Define transforms
image_transform = transforms.Compose([
                                    transforms.Resize((224,224)),         # resize shortest side to 224 pixels.
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

#========================================================================================#
#========================================================================================#



#=====================================  Modules  ========================================#
#========================================================================================#

## Insert inputs to prob varibale.
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


## Tagging & Export images taged as train/test datasets.
def Mark_Sets(Set):
    # Input: 'train' / 'test'.
    # Output: imgaes folder path , images file path.
    files_folder = os.path.join(base_path,'files')
    images_folder = f'{base_path}/data/{Set}'
    image_tagged_file = os.path.join(files_folder,f'{Set}_tags.csv')
    image_dict = {Set : sorted(os.listdir(images_folder))}
    tags_file = remo.generate_image_tags(tags_dictionary  = image_dict, 
                                         output_file_path = image_tagged_file)
    return images_folder, tags_file 

    
## Create / Add images to databse & Create train annotation. 
def CreateAddDataSet():
    try:
        my_dataset = next(filter(lambda row: row.name==dataset_name, remo.list_datasets()))
    except:
        my_dataset = remo.create_dataset(name = dataset_name,
                                         annotation_task = 'Image classification')
        remo.create_annotation_set(annotation_task='Image classification', 
                           dataset_id=my_dataset.id, 
                           name = all_annotation, 
                           classes = list(Classes.values()))
    
    ## Add images to database.
    def Add_data(Set):
        # Input: 'train' / 'test'.
        images_folder, images_marks = Mark_Sets(Set)
        annotation_id = next(filter(lambda row: row.name==all_annotation, my_dataset.annotation_sets())).id
        
        my_dataset.add_data(local_files =[images_folder, images_marks],
                            annotation_set_id = annotation_id)
    Add_data('train'); 
    Add_data('test')
    
    return my_dataset


## Update/Create train annotation csv file, after tagging train annotation dataset or 
## after fixing prediction annotation in remo app.
def Update_annotation(mode):
    # Input: 'annotate_new_data' / 'annotate_prediction'.
    try:
        database_id = next(filter(lambda row: row.name==dataset_name, remo.list_datasets())).id
        dataset = remo.get_dataset(dataset_id = database_id)
    except:
        return print(f'Error:   There is not {dataset_name} database.')
    
    main_annotaion_id,prediction_annotaion_id = None,None
    for i in dataset.annotation_sets():
        if i.name == all_annotation:
            main_annotaion_id = i.id
        if i.name == predicaion_annotation:
            prediction_annotaion_id = i.id
            
    def export_annotion(Set): # train or test
        annotation_file_path = os.path.join(f'{base_path}/files',f'annotations_{Set}.csv')
        dataset.export_annotations_to_file(annotation_file_path,
                                              annotation_set_id = main_annotaion_id,
                                              annotation_format='csv',
                                              append_path = False,
                                              export_tags = False,
                                              filter_by_tags=[Set])
    if mode=='annotate_new_data':
        export_annotion('train')
        export_annotion('test')
        
    elif mode == 'annotate_prediction':
        prediction_annotaion = dataset.annotations(annotation_set_id=prediction_annotaion_id) 
        annotations = []
        for item in prediction_annotaion:
            annotation = remo.Annotation()
            annotation.img_filename = item.img_filename
            annotation.classes=item.classes[0]
            annotations.append(annotation)
        dataset.add_annotations(annotations=annotations,annotation_set_id=main_annotaion_id)
        export_annotion('train')
    else:
        return print(f'Error:   Wrong mode.')



## Read train/test annotation csv file and return images tensors.
class Data(Dataset):
    def __init__(self,mode=None,size=5):
        self.mode = mode
        self.classes = Classes
        self.size = size
        self.transform = image_transform
        self.a = 'train' if self.mode=='prediction' else self.mode
        self.data = pd.read_csv(os.path.join(base_path,f'files/annotations_{self.a}.csv'))
        
        if self.mode == 'test' or self.mode == 'train':
            self.data = self.data.dropna()
            
        elif self.mode == 'prediction':
            self.data = self.data[self.data['classes'].isnull()]
            self.data = self.data.iloc[:size,:]
            
        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        images_path = f"{base_path}/data/{self.a}/{self.data.loc[idx, 'file_name']}"
        images = Image.open(images_path)
        
        X_set = self.transform(images)
        
        if self.mode == 'test' or self.mode == 'train':    
            image_labels = list(self.classes.values()).index(self.data.loc[idx, 'classes'])
            y_set =  torch.as_tensor(image_labels, dtype=torch.long)
            return (X_set, y_set)
        elif self.mode == 'prediction':
            return (X_set,None)


            

## Load / Create annotation model.
def load_model():
    model_folder = f'{base_path}/model'
    files = os.listdir(model_folder)
    if model_name not in files:
        ## Download model
        model = models.alexnet(pretrained=False) 
        ## Modify the classifier
        model.classifier = nn.Sequential(nn.Linear(9216, 1024), 
                                         nn.ReLU(),
                                         nn.Dropout(0.4),
                                         nn.Linear(1024, len(Classes)),
                                         nn.LogSoftmax(dim=1))
        torch.save(model, os.path.join(model_folder,model_name))
    else:
        model = torch.load(os.path.join(model_folder,model_name))
        model.eval()
    
    if len(files)<4:
        check_create_model_files(model_folder)
        
    return model

## Create files record.
def check_create_model_files(model_folder):
    if 'trained.csv' not in model_folder:
        df = pd.DataFrame()
        df['file_name'] = None
        df['classes'] = None
        df.to_csv(f'{model_folder}/trained.csv', mode='a', index=False ,header=True)  
    if 'tested.csv' not in model_folder:
        df = pd.DataFrame()
        df['file_name'] = None
        df[list(Classes.values())] = [None for i in range(len(Classes))]
        df['prediction'] = None
        df['True value'] = None
        df.to_csv(f'{model_folder}/tested.csv', mode='a', index=False ,header=True)
    if 'predicted.csv' not in model_folder:
        df = pd.DataFrame()
        df['file_name'] = None
        df[list(Classes.values())] = [None for i in range(len(Classes))]
        df['classes'] = None
        df.to_csv(f'{model_folder}/predicted.csv', mode='a', index=False ,header=True)



## Train, Test and Predicte based on model mode.
def TrainTestModel(mode,epochs=1,batch_size=1,size=5):
    # Input: mode : 'train' / 'test' / 'prediction'.
    #        size : How many images from train dataset to predict.
    dataset = Data(mode=mode,size=size)
    model = load_model()
    model.eval()
    
    if dataset.mode=='train':
        trained_data = pd.read_csv(f'{base_path}/model/trained.csv')
        dataset.data = pd.concat([dataset.data ,trained_data],axis=0)
        dataset.data.drop_duplicates(subset=['file_name'],keep=False,inplace=True)
        dataset.data.reset_index(drop=True, inplace=True)
        if len(dataset.data)==0:
            return print("Model not trained. -> No new data to train on.")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        epochs = epochs
        for i in range(epochs):
            for b, (X_train, y_train) in enumerate(dataset_loader,1):
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'epoch: {i:2}   loss: {loss.item():7.5f}')
        ## Save the trained model & what his trained on.
        torch.save(model, f'{base_path}/model/{model_name}')   
        dataset.data.to_csv(f'{base_path}/model/trained.csv', mode='a', index=False ,header=False)
    
    elif dataset.mode=='test' or dataset.mode=='prediction':
        image_name, true_val, prob, prediction=[],[],[],[]
        for i ,item in enumerate(dataset):
            with torch.no_grad():
                y_pred = model(item[0].view(1,3,224,224))
                prediction.append(Classes[torch.max(y_pred.data, 1)[1].item()])
                prob.append(list(map(lambda x: x.item(),torch.exp(y_pred.data[0]))))
                image_name.append(dataset.data.loc[i,'file_name'])
                try:
                    true_val.append(Classes[item[1].item()])
                except:
                    pass
                if i == len(dataset)-1:
                    break
        df = pd.DataFrame({'file_name':image_name})
        df[list(Classes.values())] = prob
        df['classes'] = prediction # predictions
        dataset.data = df
        if dataset.mode == 'test':
            df['True value'] = true_val
            df.to_csv(f'{base_path}/model/tested.csv', mode='a', index=False ,header=False)
        else:
            df.to_csv(f'{base_path}/model/predicted.csv', mode='a', index=False ,header=False)
            df.to_csv(f'{base_path}/files/prediction.csv', index=False)
        print(df)
        


## Upload Prediction Resul by prob.
## str check if class is less then inserted value.
## flot check if class is greater then inserted value. 
## None return class with the max probability.
def Upload_annotation(lst):
    # Input: probabilities threshold for each class (see examples).
    df1 = pd.read_csv(f'{base_path}/files/prediction.csv')
    if lst ==[]:
        try:
            df = df1.drop(columns=list(Classes.values()))
            df.to_csv(f'{base_path}/files/prediction.csv', index=False)
        except:
            pass
    else:    
        for i,item in enumerate(list(Classes.values())):
            if type(lst[i]) ==float:
                df1 = df1[df1[item]>float(lst[i])]
                df1.reset_index(drop=True, inplace=True)
            elif type(lst[i]) ==str:
                df1 = df1[df1[item]<float(lst[i])]
                df1.reset_index(drop=True, inplace=True)
        df1 = df1.drop(columns=list(Classes.values()))
        df1.to_csv(f'{base_path}/files/prediction.csv',index=False)
    
    my_dataset = next(filter(lambda row: row.name==dataset_name, remo.list_datasets()))
    predictions = my_dataset.create_annotation_set(annotation_task='Image Classification', 
                                                name = predicaion_annotation,
                                                paths_to_files = [f'{base_path}/files/prediction.csv'],
                                                classes = list(Classes.values()))
    my_dataset.view()

# Example 1. Upload_annotation([[]), return class with max probability.
# Example 2. Upload_annotation(['0.3',0.8]), return class with max probability from list of
#                                            class a<0.3 & Class b>0.8.

#========================================================================================#
#========================================================================================#

