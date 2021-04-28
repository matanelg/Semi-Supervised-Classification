import os
os.chdir('/home/matanel/Remo/Classification_Semi_Supervised/code')
from globals_var import *


@click.command()
@click.option('--mode',
              default = None,
              type=click.Choice(['create',
                                 'export_annotation',
                                 'train',
                                 'test',
                                 'prediction',
                                 'upload_annotation',
                                 'update_export_train']))
@click.option('--dataset',
              default = None,
              type=click.Choice(['train','test','']),
              help="Create / Add dataset to the database")
@click.option('--epochs',
              default = 1,
              help="(Use only on train mode)",
              type=int)
@click.option('--batch_size',
              default = 1,
              help="(Use only on train mode)",
              type=int)
@click.option('--images_size',
              default = 10,
              help='how many images to predict. (Use only on prediction mode)',
              type=int)
@click.option('--prob', 
              help='Probability threshold for each class. (Use only on upload_annotation mode)',
              cls=PythonLiteralOption, 
              default='[]')


def main(mode,dataset,epochs,batch_size,images_size,prob):
    """Operation:.

    1. Create / add database by adding images to train & test folders.
    
    2. Export annotation after tagging.
    
    3. Train model, choose epochs and batch size as you want. 

    4. Inference Test, check the probabilities for each class.
    
    5. Inference prediction, check the probabilities for each class.
    
    6. Upload Predicted annotation (now there is also predicted annotation).
    
    7. Fix predicted annotation.
    
    8. Update train annotation (true annotaion) and export file.
    
    9. Delete predicted annotaion.
    
    """
    ## Create databse or add dataset.
    if mode == 'create':
        my_dataset = CreateAddDataSet(dataset)
        my_dataset.view()
    
    ## Export tagged annotation.
    elif mode == 'export_annotation':
            Update_annotation(mode='annotate_new_data')
    
    ## Train model.
    elif mode == 'train':
        TrainTestModel(mode = 'train',
                       epochs=epochs,
                       batch_size=batch_size)
    
    ## Test model.
    elif mode == 'test':
        TrainTestModel(mode = 'test')
    
    ## Inference prediction.
    elif mode == 'prediction':
        TrainTestModel(mode = 'prediction',
                       size=images_size)
    
    ## Upload prediction annotation set.
    elif mode == 'upload_annotation':
        ## prob : list of probability threshold for each class.
        ## Example : prob = ["'0.3'",0.8] means keep all probs<0.3 for class A,
        ## and probs >0.8 for class B and then annotate.
        ## Default : None - annotate the max value class. 
        Upload_annotation(prob)

    ## Update & export tru annotation.
    elif mode == 'update_export_train':
        Update_annotation(mode='annotate_prediction')



if __name__=='__main__':
    main()






# dataset = Data(mode='train')
# model = load_model()
# model.eval()

# if dataset.mode=='train':
#     trained_data = pd.read_csv(f'{base_path}/model/trained.csv')
#     dataset.data = pd.concat([dataset.data ,trained_data],axis=0)
#     dataset.data.drop_duplicates(subset=['file_name'],keep=False,inplace=True)
#     dataset.data.reset_index(drop=True, inplace=True)
#     if len(dataset.data)==0:
#         print("Model not trained. -> No new data to train on.")
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     epochs = epochs
#     for i in range(epochs):
#         for b, (X_train, y_train) in enumerate(dataset_loader,1):
#             y_pred = model(X_train)
#             loss = criterion(y_pred, y_train)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             print(f'epoch: {i:2}   loss: {loss.item():7.5f}')
#     ## Save the trained model & what his trained on.
#     torch.save(model, f'{base_path}/model/{model_name}')   
#     dataset.data.to_csv(f'{base_path}/model/trained.csv', mode='a', index=False ,header=False)

# elif dataset.mode=='test' or dataset.mode=='prediction':
#     image_name, true_val, prob, prediction=[],[],[],[]
#     for i ,item in enumerate(dataset):
#         with torch.no_grad():
#             y_pred = model(item[0].view(1,3,224,224))
#             prediction.append(Classes[torch.max(y_pred.data, 1)[1].item()])
#             prob.append(list(map(lambda x: float(f'{x.item():.3f}'),torch.exp(y_pred.data[0])))) #==#
#             image_name.append(dataset.data.loc[i,'file_name'])
#             try:
#                 true_val.append(Classes[item[1].item()])
#             except:
#                 pass
#             if i == len(dataset)-1:
#                 break
#     df = pd.DataFrame({'file_name':image_name})
#     df[list(Classes.values())] = prob
#     df['classes'] = prediction # predictions
#     dataset.data = df
#     if dataset.mode == 'test':
#         df['True value'] = true_val
#         df.to_csv(f'{base_path}/model/tested.csv', mode='a', index=False ,header=False)
#     else:
#         df.to_csv(f'{base_path}/model/predicted.csv', mode='a', index=False ,header=False)
#         df.to_csv(f'{base_path}/files/prediction.csv', index=False)
#     print(df)