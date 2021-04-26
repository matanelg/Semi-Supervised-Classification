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
# Create/Add data from train & test folders to remo app.
```python
python main.py --mode=create
```
### Tag your images 

### Export annotation
```python
python main.py --mode=export_annotation
```

### Train model
```python
python main.py --mode=train
```

### Test model
```python
python main.py --mode=test
```

### Predict untagged images in train folder
```python
python main.py --mode=inference --images_size=5
```

### Upload new annotation set
```python
python main.py --mode=upload_annotation
```

### Fix images tagging

### Update annotation
```python
python main.py --mode=update_export_train
```

### Delete annotation set








