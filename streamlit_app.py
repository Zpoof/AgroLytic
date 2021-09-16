import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import requests, json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

st.image(Image.open('/content/drive/MyDrive/epic.png'))
tab = st.sidebar.radio("Navigation", ['SoilID','CropChoice'])
model_path = "/content/drive/MyDrive/SoilNet_93_86.h5"
SoilNet = load_model(model_path)
classes = {0:"Alluvial Soil:-{ Rice,Wheat,Sugarcane,Maize,Cotton,Soyabean,Jute }",1:"Black Soil:-{ Virginia, Wheat , Jowar,Millets,Linseed,Castor,Sunflower} ",2:"Clay Soil:-{ Rice,Lettuce,Chard,Broccoli,Cabbage,Snap Beans }",3:"Red Soil:{ Cotton,Wheat,Pilses,Millets,OilSeeds,Potatoes }"}
crops = ["Rice  \nWheat  \nSugarcane  \nMaize  \nCotton  \nSoyabean  \nJute", 
"Wheat  \nJowar  \nMillets  \nLinseed  \nCastor  \nSunflower", "Rice  \nLettuce  \nChard  \nBroccoli  \nCabbage  \nSnap Beans", "Cotton  \nWheat  \nPulses  \nMillets  \nOil Seeds  \nPotatoes"]
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
    pred,res = model_predict(image, SoilNet)
    st.header("Predicted type: " + pred + " Soil")
    st.subheader("Crop Reccomendations:   \n" + crops[res])
else if tab == "CropChoice"
  st.title("CropChoice")
  st.header("Find the best crop to grow for your land")
  st.write("")
  
