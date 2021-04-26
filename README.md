# Auto Semi Supervised Labeling

This project preforms auto annotation for classification problem using by semi supervised learning.<br />
The main goal is to generate tagged database for later use of a model training.<br />
We will be use [remo](https://remo.ai/docs/) package python library for annotation and images display gui.<br />

## Installation

1. Create python environment using the requirements.txt file.<br />

2. You can follow [remo](https://remo.ai/docs/) guide installation or<br />
```bash
pip install remo
python -m remo_app init
python -m remo_app
```
3. Clone repository and move to code folder.
```bash
git clone https://github.com/matanelg/Semi-Supervised-Classification.git
cd ./code
```

## Operation
### 1. Create/Add data from train & test folders to remo app.
```python
python main.py --mode=create
```
This command also create test_tags and train_tags csv files that remo read and classified in app.

### 2. Tag your images 

### 3. Export annotation
```python
python main.py --mode=export_annotation
```
Save all tagged images in annotations_test & annotations_train csv files.

### 4. Train model
```python
python main.py --mode=train
```
You can also use --epochs and --batch_sise options the defualt is 1.

### 5. Test model
```python
python main.py --mode=test
```
Show probabilities of the classes.

### 6. Predict untagged images in train folder
```python
python main.py --mode=inference --images_size=5
```
Show predictions and probabilities.

### 7. Upload new annotation set
```python
python main.py --mode=upload_annotation
```
Create new annotation of prediction set and upload to remo app. 

### 8. Fix images wrong tag in remo app

### 9. Update annotation
```python
python main.py --mode=update_export_train
```
Add the correct new tags to annotation_train.csv

### 10. Delete annotation set
Keep tagging your data and return this loop until<br/>
your model will be accurate enough to tagg the rest images by his own. 








