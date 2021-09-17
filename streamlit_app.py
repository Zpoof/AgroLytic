import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import requests, json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.image(Image.open('epic.png'))
tab = st.sidebar.radio("Navigation", ['Home','SoilID','CropChoice'])
model_path = "SoilNet_93_86.h5"

SoilNet = load_model(model_path)
CropChoice = joblib.load("CropChoice.joblib.pkl")

classes = {0:"Alluvial Soil:-{ Rice,Wheat,Sugarcane,Maize,Cotton,Soyabean,Jute }",1:"Black Soil:-{ Virginia, Wheat , Jowar,Millets,Linseed,Castor,Sunflower} ",2:"Clay Soil:-{ Rice,Lettuce,Chard,Broccoli,Cabbage,Snap Beans }",3:"Red Soil:{ Cotton,Wheat,Pilses,Millets,OilSeeds,Potatoes }"}
crops = ["Rice  \nWheat  \nSugarcane  \nMaize  \nCotton  \nSoyabean  \nJute", 
"Wheat  \nJowar  \nMillets  \nLinseed  \nCastor  \nSunflower", "Rice  \nLettuce  \nChard  \nBroccoli  \nCabbage  \nSnap Beans", "Cotton  \nWheat  \nPulses  \nMillets  \nOil Seeds  \nPotatoes"]

choices = ['wheat','mungbean','Tea','millet','maize','lentil','jute','cofee','cotton','ground nut','peas','rubber','sugarcane','tobacco','kidney beans','moth beans','coconut','blackgram','adzuki beans','pigeon peas','chick peas','banana','grapes','apple','mango','muskmelon','orange','papaya','watermelon','pomegranate']

data=pd.read_csv('cpdata.csv')
label= pd.get_dummies(data.label).iloc[: , 1:]
data= pd.concat([data,label],axis=1)
data.drop('label', axis=1,inplace=True)
train=data.iloc[:, 0:4].values
test=data.iloc[: ,4:].values

X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.3)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def model_predict(image,model):
    print("Predicted")
    #image = load_img(image_path,target_size=(224,224))
    image = image.resize((224,224))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = np.argmax(model.predict(image))
    prediction = classes[result]
    
    if result == 0:
        #st.header("Alluvial Soil")
        
        return "Alluvial", result
    elif result == 1:
        #st.header("Black Soil")
        
        return "Black", result
    elif result == 2:
        #st.header("Clay Soil")
        
        return "Clay", result
    elif result == 3:
        #st.header("Red Soil")
        
        return "Red", result

if tab == "SoilID":
  st.title("SoilID")
  st.header("Identify your soil type and get intelligent crop reccomendations")
  st.write("")
  uploaded_file = st.file_uploader("Upload image of soil")
  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)
    pred,res = model_predict(image, SoilNet)
    st.header("Predicted type: " + pred + " Soil")
    st.subheader("Crop Reccomendations:   \n" + crops[res])
elif tab == "Home":
  st.title("Home")
  st.header("Welcome to your Agrolytic")
  st.map()
  st.header("//Weather Info Here")
    
elif tab == "CropChoice":
  st.title("CropChoice")
  st.header("Find the best crop to grow for your land")
  st.write("")

  x = []
  temp = st.slider("Temperature", min_value = 0.0, max_value = 50.0, value = 20.0, step = 0.2)
  humid = st.slider("Humidity", min_value = 0.0, max_value =100.0, value = 50.0, step = 0.2)
  ph = st.slider("Ph", min_value = 0.0, max_value = 14.0, value = 5.0, step = 0.1)
  rainfall = st.slider("Annual Rainfall", min_value = 0.0, max_value = 400.0, value = 100.0, step = 1.0)
  x.append(temp)
  x.append(humid)
  x.append(ph)
  x.append(rainfall)
  if st.button("Find Crop"):
    pred = CropChoice.predict(sc.transform([x]))
    pred = pred.tolist()[0]
    for n in range(len(pred)):
      if pred[n] == 1:
        st.header("Optimal Crop: " + choices[n].title())

