# Packages 
import os
import requests
import streamlit as st
from requests.exceptions import RequestException, Timeout


# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
TIMEOUT = 2  # secondes

st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="centered"
)


# Header
st.title("Breast Cancer Prediction App")

st.markdown(
    """
    <div style="text-align: center;">
        Cette application calcule la probabilité de développer un cancer du sein.
    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# Features (ordre strict)
FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
]


# Formulaire
st.subheader("Paramètres d’entrée")

inputs = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(FEATURES):
    column = col1 if i % 2 == 0 else col2
    with column:
        inputs[feature] = st.number_input(
            label=feature.replace("_", " ").capitalize(),
            min_value=0.0001,         
            value=0.1,                
            step=0.0001,
            format="%.5f",
            help=f"Valeur positive requise pour {feature}"
        )


# Validation locale
def validate_inputs(data: dict) -> bool:
    return all(v > 0 for v in data.values())

# Prediction
st.divider()

if st.button("Résultats", use_container_width=True):

    if not validate_inputs(inputs):
        st.warning("Toutes les valeurs doivent être strictement supérieures à 0.")
        st.stop()

    with st.spinner(" Analyse en cours..."):
        try:
            response = requests.post(
                API_URL,
                json=inputs,
                timeout=TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()

                st.markdown("### Résultat de la prédiction")

                # Signal visuel selon le diagnostic
                if "Bénin" in result["label"]:
                    st.success(" **Aucune anomalie détectée**")
                else:
                    st.error(" **Présence probable d’une tumeur maligne**")

               
                st.markdown(
                    f"""
                    <div style="text-align: center; font-size: 1.2rem;">
                        <strong>Diagnostic :</strong><br>
                        {result["label"]}
                        <br><br>
                        <strong>Probabilité estimée :</strong><br>
                        <span style="font-size: 1.6rem;">
                            {result["probability"] * 100:.2f}%
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption( " Cette prédiction ne remplace pas un avis médical.")


            elif response.status_code == 422:
                st.error("Données invalides envoyées à l’API")
                st.json(response.json())

            else:
                st.error("Erreur côté API")
                st.write(f"Code HTTP : {response.status_code}")
                st.json(response.json())

        except Timeout:
            st.error("L’API ne répond pas.")

        except RequestException as e:
            st.error("Erreur réseau lors de l’appel API")
            st.code(str(e))

        except Exception as e:
            st.error("Erreur inattendue")
            st.code(str(e))
