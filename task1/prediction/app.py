import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Charger le modèle entraîné
model = joblib.load('model_predict1.pkl')

# Vérification que le modèle est bien un modèle scikit-learn
if not hasattr(model, 'predict'):
    st.error("Le fichier chargé n'est pas un modèle valide.")
    st.stop()

# Titre de l'application
st.title("Prédiction de survie sur le Titanic")

# Afficher une image du Titanic
image = Image.open('titanic.jpeg')
st.image(image, caption='RMS Titanic', use_column_width=True)

# Entrée des données utilisateur
st.sidebar.header('Entrer les caractéristiques du passager :')

def user_input_features():
    Pclass = st.sidebar.selectbox('Classe', (1, 2, 3))
    Sex = st.sidebar.selectbox('Sexe', ('Homme', 'Femme'))
    Age = st.sidebar.slider('Âge', 0, 80, 22)
    Cabin = st.sidebar.selectbox('Cabine', (1, 2, 3, 4, 5, 6, 7))
    Q = st.sidebar.selectbox('Embarked (Q)', (0, 1))
    S = st.sidebar.selectbox('Embarked (S)', (0, 1))

    data = {'Pclass': Pclass,
            'Sex': 0 if Sex == 'Homme' else 1,
            'Age': Age,
            'Cabin': Cabin,
            'Q': Q,
            'S': S}
    features = np.array([list(data.values())])
    return features

input_df = user_input_features()

# Prédiction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Affichage des résultats
st.subheader('Résultats de la prédiction')
if prediction[0] == 1:
    st.write("### Ce passager aurait survécu.")
    st.write(f"Il y a une probabilité de survie de {prediction_proba[0][1]:.2%}.")
else:
    st.write("### Ce passager n'aurait pas survécu.")
    st.write(f"Il y a une probabilité de survie de {prediction_proba[0][1]:.2%}.")
