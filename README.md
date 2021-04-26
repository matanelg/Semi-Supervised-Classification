# Auto Semi Supervised Labeling

This project preforms auto annotation for classification problem using by semi supervised learning.__
The main goal is to generate tagged database for later use of a model training.__
We will be use [remo](https://remo.ai/docs/) package python library for annotation and images display gui.__

## Installation

### 1. Step 1
Create python environment using the requirements.txt file.__

### 2. Step 2
You can follow [remo](https://remo.ai/docs/) guide installation or__

'''bash
pip install remo
python -m remo_app init
'''

### 3. Step 3
Set up the viewer in the Config file (at your home directory .remo/remo.json)
for one of electron/browser/jupyter.__
You can also register and connect with user name and password to the service.__

### 4. Step 4
'''python
python -m remo_app
'''


