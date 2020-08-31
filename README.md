# Medico
Medico is an AI model which assess the health of an apple leaf and classifies to one of the four categories
1. Healthy Leaf
2. Apple Scab
3. Apple Rust
4. Multiple Disease

<img src="src/img/screenshot1.png?raw=true" width="1000">
<img src="src/img/screenshot2.png?raw=true" width="1000">

### How to Run:
1. Install necessary modules with `sudo pip3 install -r requirements.txt` command.
2. Go to __src__ folder (if you want to change paths of files and folders, go to _**src/config.py**_).
3. Run `python3 train.py` to train and save the machine learning model.
4. To run this app from **Streamlit**. Run `streamlit run streamlitapp.py`.
5. Upload your Plant Leaf Image and then check the result.

### Data
Data can be downloaded from [here](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data)

> Web App was made using [__Streamlit__](https://www.streamlit.io/)

__Please Give a :star2: if you :+1: it.__
