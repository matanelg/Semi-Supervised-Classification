# Auto Semi Supervised Labeling

This project preforms auto annotation for classification problem using by semi supervised learning.<br />
The main goal is to generate tagged database for later use of a model training.<br />
We will be use [remo](https://remo.ai/docs/) package python library for annotation and images display gui.<br />

## Installation

### 1. Step 1
Create python environment using the requirements.txt file.<br />

### 2. Step 2
You can follow [remo](https://remo.ai/docs/) guide installation or<br />

```bash
pip install remo
python -m remo_app init
```

### 3. Step 3
Set up the viewer in the Config file (at your home directory .remo/remo.json)
for one of electron/browser/jupyter.<br />
You can also register and connect with user name and password to the service.<br />

### 4. Step 4
```python
python -m remo_app
```


