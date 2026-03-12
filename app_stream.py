"""
Interface Streamlit pour la prédiction du cancer du sein
Design moderne avec palette de couleurs santé
"""

import os
import requests
import streamlit as st
from requests.exceptions import RequestException, Timeout

# Configuration en local 
# API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# Configuration en conteneur
API_URL = os.getenv("API_URL", "http://host.docker.internal:8000/predict")
TIMEOUT = 15

st.set_page_config(
    page_title="Prédiction Cancer du Sein",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    
    h1 {
        color: #0d6efd;
        text-align: center;
        padding: 1rem 0;
    }
    
    h3 {
        color: #495057;
        border-left: 4px solid #0d6efd;
        padding-left: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #0d6efd 0%, #0056b3 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(13, 110, 253, 0.4);
        transform: translateY(-2px);
    }
    
    .stNumberInput > div > div > input {
        border-radius: 6px;
        border: 1px solid #ced4da;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# En-tête
st.markdown("""
    <div class="info-box">
        <h1>🩺 Prédiction du Cancer du Sein</h1>
        <p style="margin: 0; font-size: 1.1rem;">
            Outil d'aide au diagnostic basé sur l'Intelligence Artificielle
        </p>
    </div>
""", unsafe_allow_html=True)

# Configuration des features
FEATURES = [
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

FEATURE_LABELS = {
    "radius_worst": "Radius (valeur maximale)",
    "texture_worst": "Texture (valeur maximale)",
    "perimeter_worst": "Perimeter (valeur maximale)",
    "area_worst": "Area (valeur maximale)",
    "smoothness_worst": "Smoothness (valeur maximale)",
    "compactness_worst": "Compactness (valeur maximale)",
    "concavity_worst": "Concavity (valeur maximale)",
    "concave_points_worst": "Concave points (valeur maximale)",
    "symmetry_worst": "Symmetry (valeur maximale)",
    "fractal_dimension_worst": "Fractal dimension (valeur maximale)"
}

# Formulaire de saisie
with st.expander(" À propos des paramètres", expanded=False):
    st.info(
        "Ces paramètres correspondent aux caractéristiques les plus critiques "
        "des cellules tumorales, mesurées à partir d'images de biopsie. "
        "Ces paramètres prennent des valeurs comprises entre 0 et 1."
    )

inputs = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(FEATURES):
    column = col1 if i % 2 == 0 else col2
    with column:
        inputs[feature] = st.number_input(
            label=FEATURE_LABELS[feature],
            min_value=0.0001,
            value=0.1,
            step=0.0001,
            format="%.4f",
            key=feature
        )

# Validation des données
def validate_inputs(data: dict) -> bool:
    """Vérifie que toutes les valeurs sont strictement positives"""
    return all(v > 0 for v in data.values())

# Prédiction
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Lancer l'analyse", use_container_width=True):
    
    if not validate_inputs(inputs):
        st.warning("Toutes les valeurs doivent être strictement positives.")
        st.stop()
    
    with st.spinner("🔄 Analyse en cours..."):
        try:
            response = requests.post(API_URL, json=inputs, timeout=TIMEOUT)
            
            if response.status_code == 200:
                result = response.json()
                
                st.markdown("---")
                
                # Déterminer le type de diagnostic
                is_benign = result["prediction"] == "B"
                
                if is_benign:
                    badge_color = "#28a745"
                    icon_color = "#155724"
                    bg_color = "#d4edda"
                    status_text = "NÉGATIF"
                    diagnosis = "Absence de cancer"
                else:
                    badge_color = "#dc3545"
                    icon_color = "#721c24"
                    bg_color = "#f8d7da"
                    status_text = "POSITIF"
                    diagnosis = "Présence de cancer"
                
                # Sortie professionnelle minimaliste
                st.markdown(f"""
                    <div style="background: white; border-radius: 16px; padding: 3rem 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08); text-align: center; max-width: 500px; margin: 2rem auto;">
                        <div style="width: 80px; height: 80px; background: {bg_color}; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin: 0 auto 1.5rem;">
                            <div style="font-size: 2.5rem; color: {icon_color};">
                                {'✓' if is_benign else '!'}
                            </div>
                        </div>
                        <div style="font-size: 2rem; color: #6c757d; letter-spacing: 2px; font-weight: 600; margin-bottom: 0.5rem;">
                            RÉSULTAT DU TEST
                        </div>
                        <h2 style="color: {badge_color}; font-size: 1.8rem; font-weight: 600; margin: 0 0 0.5rem 0;">
                            {status_text}
                        </h2>
                        <p style="color: #6c757d; font-size: 1.5rem; margin: 0 0 2rem 0;">
                            ({diagnosis})
                        </p>
                        <div style="width: 60px; height: 2px; background: #dee2e6; margin: 2rem auto;"></div>
                        <div style="font-size: 2rem; color: #adb5bd; letter-spacing: 1.0px; font-weight: 500; margin-bottom: 0.5rem;">
                            Score
                        </div>
                        <div style="font-size: 2rem; font-weight: 400; color: {badge_color}; line-height: 1;">
                            {result["probability"] * 100:.1f}<span style="font-size: 2rem; color: #6c757d;">%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                
                # Avertissement médical
                st.warning(
                    "**Important :** Ce résultat est fourni à titre indicatif uniquement. "
                    "Il ne remplace en aucun cas l'avis d'un professionnel de santé qualifié. "
                    "Consultez un médecin pour un diagnostic définitif."
                )
            
            elif response.status_code == 422:
                st.error("Données invalides")
                with st.expander("Détails de l'erreur"):
                    st.json(response.json())
            
            else:
                st.error(f"Erreur API (Code {response.status_code})")
                with st.expander("Détails de l'erreur"):
                    st.json(response.json())
        
        except Timeout:
            st.error(
                "**Délai d'attente dépassé**\n\n"
                "L'API ne répond pas. Vérifiez que le serveur est démarré."
            )
        
        except RequestException as e:
            st.error("**Erreur de connexion**")
            with st.expander("Détails techniques"):
                st.code(str(e))
        
        except KeyError as e:
            st.error(f"**Champ manquant dans la réponse API** : {e}")
            with st.expander("Réponse reçue"):
                st.json(response.json() if 'response' in locals() else {})
        
        except Exception as e:
            st.error("**Erreur inattendue**")
            with st.expander("Détails techniques"):
                st.code(f"{type(e).__name__}: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
        <div style="text-align: center; color: #6c757d; padding: 1rem;">
            <small>
                🔒 Vos données sont traitées de manière confidentielle et ne sont pas stockées.<br>
                Version 1.0.0 | Propulsé par FastAPI + Streamlit + Docker
            </small>
        </div>
    """, unsafe_allow_html=True)


