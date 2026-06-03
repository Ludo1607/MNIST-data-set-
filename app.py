
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Charger le modèle
cnn = load_model("mon_modele_mnist.h5")

st.title("🔢 Reconnaissance de chiffres manuscrits")
st.write("Téléverse une image contenant un chiffre manuscrit pour prédiction")

uploaded_file = st.file_uploader("Choisir une image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)
    
    # Prétraitement
    img = image.convert('L')
    img = img.resize((28,28))
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1,28,28,1)
    
    # Prédiction
    prediction = cnn.predict(img_array)
    chiffre = np.argmax(prediction)
    confiance = np.max(prediction) * 100
    
    st.success(f"Chiffre prédit : {chiffre}")
    st.info(f"Confiance : {confiance:.2f}%")
