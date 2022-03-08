from xml.parsers.expat import model
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import os
from joblib import dump, load
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle

# Titulo do app
st.write("""
# Prevendo ocorrência de doenças cardio vasculares

 Aplicativo que utiliza Machine Learning para prever possíveis ocorrências de doenças do coração. \n

 Fonte dos dados: www.kaglle.com \n

 Legenda dos atributos:\n

 Gênero: 1 - Feminino  2 - Masculino /
 Colesterol: 1 - Normal  2 - Acima do normal  3 - Bem acima do normal /
 Glicose: 1 - Normal  2 - Acima do normal  3 - Bem acima do normal /
 Fumante: 0 - Não  1 - Sim /
 Bebiba Alcoólica: 0 - Não  1 - Sim /
 Praticante de atividade física: 0 - Não  1 - Sim 
""")


# Cabeçalho
st.subheader("Características do Paciente")

# Nome do paciente
user_input = st.sidebar.text_input("Digite seu nome")

st.write("Paciente: ", user_input)

# Dados do paciente
def user_input_features():
    gender = st.sidebar.selectbox("Gênero", [1, 2], 1)
    age = st.sidebar.slider("Idade", 15, 65, 20)
    height = st.sidebar.slider("Altura", 0.5, 2.5, 1.70)
    weight = st.sidebar.slider("Peso", 10, 200, 60)
    ap_hi = st.sidebar.slider("Pressão sistólica", 0, 500, 100)
    ap_lo = st.sidebar.slider("Pressão diastólica", 0, 500, 100)
    cholesterol = st.sidebar.selectbox("Colesterol", [1,2 ,3], 1)
    gluc = st.sidebar.selectbox("Glicose",[1,2, 3] ,1)
    smoke = st.sidebar.selectbox("Fumante", [0, 1], 0)
    alco = st.sidebar.selectbox("Bebida Alcoolica", [0 ,1], 0)
    active = st.sidebar.selectbox("Praticante de atividade física", [0, 1], 0)

    data = {"Gênero": gender,
            "Idade": age,
            "Altura": height,
            "Peso": weight,
            "Pressão sistólica": ap_hi,
            "Pressão diastólica": ap_lo,
            "Colesterol": cholesterol,
            "Glicose": gluc,
            "Fumante": smoke,
            "Bebida Alcoolica": alco,
            "Praticante de atividade física": active}

    features = pd.DataFrame(data, index = [0])
    
    return features

user_input_variables = user_input_features()


# Carregando os dados
cardio = pd.read_csv('cardio_app2.csv')
cardio_x = cardio.drop(columns=['cardio'], axis = 1)
df = pd.concat([user_input_variables, cardio_x], axis = 0)

# Tabela com os dados do usuario
st.subheader('Dados do paciente')
st.write(user_input_variables)


# Função para construção do gráfico
def impPlot(imp, name):
    figure = px.bar(imp,
                    x=imp.values,
                    y=imp.keys(), labels = {'x':'Importance Value', 'index':'Columns'},
                    text=np.round(imp.values, 2),
                    title=name + ' Feature Selection Plot',
                    width=900, height=600)
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    st.plotly_chart(figure)


# Carregando o modelo
modelo_clf = pickle.load(open('model_randomforest.pkl', 'rb'))


# Gráfico feat_importances
feat_importances = pd.Series(modelo_clf.feature_importances_, index=cardio_x.columns).sort_values(ascending=True)
st.write(impPlot(feat_importances, 'Random Forest Classifier'))

# Apply model to make predictions
prediction = modelo_clf.predict(user_input_variables)
prediction_proba = modelo_clf.predict_proba(user_input_variables)

st.subheader("Previsão: ")
st.write("""
#0 -  Não\n
#1 - Sim

#""")

st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
