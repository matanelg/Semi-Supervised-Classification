
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

