# -*- coding: utf-8 -*-
"""
Application de détection de faux billets - Version optimisée pour Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import base64
import io
import os
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Détection de Faux Billets",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Chargement du modèle ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_model.sav")
        scaler = joblib.load("scaler.sav")
        return model, scaler
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {str(e)}")
        return None, None

model, scaler = load_model()

# Chemins des images (fichiers à la racine)
GENUINE_BILL_IMAGE = "vraibillet.PNG"
FAKE_BILL_IMAGE = "fauxbillet.png"

# Vérification que les images existent
if not os.path.exists(GENUINE_BILL_IMAGE):
    st.error(f"Image manquante: {GENUINE_BILL_IMAGE}")
if not os.path.exists(FAKE_BILL_IMAGE):
    st.error(f"Image manquante: {FAKE_BILL_IMAGE}")

# Fonction pour convertir image en base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# CSS personnalisé
st.markdown("""
<style>
    :root {
        --primary: #4a6fa5;
        --secondary: #166088;
        --accent: #4fc3f7;
        --success: #4CAF50;
        --danger: #F44336;
        --light: #f8f9fa;
        --dark: #212529;
    }
   
    .header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
   
    .card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        padding: 1rem;
        margin-bottom: 1rem;
        
    }
   
    .genuine-card {
        border-left: 4px solid var(--success);
        border-radius: 25px;
    }
   
    .fake-card {
        border-left: 4px solid var(--danger);
        border-radius: 25px;
    }
   
    .stat-card {
        text-align: center;
        padding: 0.8rem;
        
    }
   
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--secondary);
    }
   
    .show-more-btn {
        margin-top: 0.5rem;
        background: #f0f2f6 !important;
        color: var(--secondary) !important;
        border: 1px solid #ddd !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h2 style="color:white; margin:0;">  📇 Application Scan Franc cfa </h2>
    <p style="color:white; opacity:0.9; margin:10;"> ✅ Solution de détection de faux billets</p>
</div>
""", unsafe_allow_html=True)

# Section Analyse  <p style="color:white; opacity:0.9; margin:10;"> 🔎💰💵 ⛶ Solution de détection de faux billets</p>
uploaded_file = st.file_uploader(
    "Faites glisser et déposez le fichier ici ou cliquez sur le bouton 'Browse files' pour Parcourir",
    type=["csv"],
    key="file_uploader"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';')
       
        # Aperçu des données
        st.markdown("#### Affichage des données reçues")
        preview_rows = 5
        table_placeholder = st.empty()
        table_placeholder.dataframe(df.head(preview_rows), height=210, use_container_width=True)
       
        if len(df) > preview_rows:
            if st.button("Afficher plus", key="show_more_btn", type="secondary"):
                table_placeholder.dataframe(df, height=min(800, len(df)*35), use_container_width=True)
       
        if st.button("Lancer la détection", key="analyze_btn"):
            with st.spinner("Analyse en cours..."):
                if model is None:
                    st.error("Modèle non chargé - Impossible d'effectuer la prédiction")
                else:
                    try:
                        # Exemple de prétraitement (à adapter selon vos colonnes)
                        required_cols = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
                        if not all(col in df.columns for col in required_cols):
                            raise ValueError("Colonnes requises manquantes dans le fichier CSV")
                           
                        features = df[required_cols]
                        features_scaled = scaler.transform(features)
                        probas = model.predict_proba(features_scaled)
                       
                        predictions = [{
                            'id': i,
                            'prediction': "Genuine" if p[1] > 0.5 else "Fake",
                            'probability': p[1]
                        } for i, p in enumerate(probas)]
                       
                        st.success("Analyse terminée avec succès !")
                       
                        # Affichage des résultats
                        st.markdown("#### Résultats de la détection")
                        genuine_img = image_to_base64(GENUINE_BILL_IMAGE)
                        fake_img = image_to_base64(FAKE_BILL_IMAGE)
                       
                        cols_per_row = 3
                        for i in range(0, len(predictions), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j in range(cols_per_row):
                                if i + j < len(predictions):
                                    pred = predictions[i + j]
                                    with cols[j]:
                                        if pred['prediction'] == 'Genuine':
                                            st.markdown(f"""
                                            <div class="card genuine-card">
                                                <div style="display:flex; align-items:center;">
                                                    <div style="flex:1;">
                                                        <h5 style="color:var(--success); margin:0 0 0.3rem 0;">Le billet n°{pred['id']} est Vrai </h5>
                                                        <p style="margin:0 0 0.2rem 0;">Probabilité: <strong>{pred['probability']*100:.1f}%</strong></p>
                                                        <div style="height:6px; background:#e9ecef; border-radius:3px;">
                                                            <div style="height:100%; width:{pred['probability']*100}%; background:var(--success); border-radius:3px;"></div>
                                                        </div>
                                                    </div>
                                                    <div style="margin-left:1rem;">
                                                        <img src="data:image/png;base64,{genuine_img}" width="80"  height="120" style="border-radius:6px;">
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"""
                                            <div class="card fake-card">
                                                <div style="display:flex; align-items:center;">
                                                    <div style="flex:1;">
                                                        <h5 style="color:var(--danger); margin:0 0 0.3rem 0;">Le billet n°{pred['id']} est Faux</h5>
                                                        <p style="margin:0 0 0.2rem 0;">Probabilité: <strong>{(1-pred['probability'])*100:.1f}%</strong></p>
                                                        <div style="height:6px; background:#e9ecef; border-radius:3px;">
                                                            <div style="height:100%; width:{(1-pred['probability'])*100}%; background:var(--danger); border-radius:3px;"></div>
                                                        </div>
                                                    </div>
                                                    <div style="margin-left:1rem;">
                                                        <img src="data:image/png;base64,{fake_img}" width="80" height="120" style="border-radius:6px;">
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                       
                        # Statistiques
                        genuine_count = sum(1 for p in predictions if p['prediction'] == 'Genuine')
                        fake_count = len(predictions) - genuine_count
                       
                        st.markdown("<h4 style='text-align: center;'>Statistiques de détection</h4>", unsafe_allow_html=True)
                       
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="card stat-card">
                                <div class="stat-value">{len(predictions)}</div>
                                <div class="stat-label">Billets analysés</div>
                            </div>
                            """, unsafe_allow_html=True)
                       
                        with col2:
                            st.markdown(f"""
                            <div class="card stat-card">
                                <div class="stat-value" style="color:var(--success);">{genuine_count}</div>
                                <div class="stat-label">Vrais billets</div>
                            </div>
                            """, unsafe_allow_html=True)
                       
                        with col3:
                            st.markdown(f"""
                            <div class="card stat-card" >
                                <div class="stat-value" style="color:var(--danger);">{fake_count}</div>
                                <div class="stat-label">Faux billets</div>
                            </div>
                            """, unsafe_allow_html=True)
                       
                        
                        st.markdown("<h4 style='text-align: center;'>Graphique de la détection</h4>", unsafe_allow_html=True)

                        fig = px.bar(
                            x=['Vrai', 'Faux'],
                            y=[genuine_count, fake_count],
                            color=['Vrai', 'Faux'],
                            color_discrete_map={'Vrai': '#4CAF50', 'Faux': '#F44336'},
                            labels={'x': 'Véracité', 'y': 'Nombre de billets'},
                            text=[genuine_count, fake_count],
                            width=450,  # Largeur légèrement augmentée pour meilleure lisibilité
                            height=500
                        )
                        
                        fig.update_traces(
                            texttemplate='%{text}',
                            textposition='outside',
                            width=0.5  # Réduction de la largeur des barres pour espacement
                        )
                        
                        fig.update_layout(
                            showlegend=True,  # Supprimé pour plus de sobriété (les couleurs sont explicites)
                            yaxis_title="Nombre de billets",
                            margin=dict(l=20, r=20, t=40, b=20),
                            autosize=False
                        )
                        
                        # Solution de centrage élégante
                        

                        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)  # Garder False pour taille fixe
                        st.markdown("</div>", unsafe_allow_html=True)
                       
                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction : {str(e)}")
    except Exception as e:
        st.error(f"Erreur de lecture du fichier: {str(e)}")
        st.markdown("""
        <div class="card" style="border-left: 4px solid var(--danger);">
            <h5>Format de fichier incorrect</h5>
            <p>Assurez-vous que votre fichier :</p>
            <ul>
                <li>Est un CSV valide</li>
                <li>Utilise le point-virgule (;) comme séparateur</li>
                <li>Contient les colonnes requises</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

