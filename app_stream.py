##=============================================================================##
##                OncoScan AI — Interface Streamlit                            ##
##                  Prédiction du cancer du sein                               ##
##=============================================================================##


## Application construite à partir d'un prototype et une amélioration avec claude IA.


# PACKAGES
import os
import hashlib
import requests
import streamlit as st
from requests.exceptions import RequestException, Timeout
import psycopg2

# CONFIGURATION
API_URL = os.getenv("API_URL", "http://api:8000/predict")
TIMEOUT = 15

DB_CONFIG = {
    "host"    : os.getenv("DB_HOST", "postgres"),
    "port"    : int(os.getenv("DB_PORT", "5432")),
    "dbname"  : os.getenv("DB_NAME",     "oncoscan"),
    "user"    : os.getenv("DB_USER",     "oncoscan"),
    "password": os.getenv("DB_PASSWORD", "oncoscan"),
}

FEATURES = [
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]

FEATURE_LABELS = {
    "radius_worst"           : "Rayon",
    "texture_worst"          : "Texture",
    "perimeter_worst"        : "Périmètre",
    "area_worst"             : "Aire",
    "smoothness_worst"       : "Régularité",
    "compactness_worst"      : "Compacité",
    "concavity_worst"        : "Concavité",
    "concave_points_worst"   : "Points concaves",
    "symmetry_worst"         : "Symétrie",
    "fractal_dimension_worst": "Dimension fractale",
}

# PAGE CONFIG
st.set_page_config(
    page_title="OncoScan AI",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Inter:wght@300;400;500;600&display=swap');

#MainMenu, footer, header, .stDeployButton { display: none !important; }

:root {
    --blue-dark  : #1e3a8a;
    --blue-mid   : #1d4ed8;
    --blue-light : #eff6ff;
    --surface    : #f1f5f9;
    --card       : #ffffff;
    --border     : rgba(30,58,138,0.10);
    --border-md  : rgba(30,58,138,0.18);
    --text-main  : #0f172a;
    --text-muted : #64748b;
    --green      : #15803d;
    --green-bg   : #f0fdf4;
    --red        : #b91c1c;
    --red-bg     : #fef2f2;
    --amber-bg   : #fffbeb;
    --amber-bdr  : #fde68a;
    --amber-text : #78350f;
}

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
section[data-testid="stAppViewContainer"] > div,
.main, .block-container {
    background-color: var(--surface) !important;
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    max-width   : 200px;
    padding-top : 0 !important;
    padding-bottom: 3rem;
}

.font-serif { font-family: 'Playfair Display', 'Times New Roman', serif; }

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background    : #f8fafc !important;
    border        : 1.5px solid #e2e8f0 !important;
    border-radius : 8px !important;
    padding       : 0.6rem 0.9rem !important;
    font-family   : 'Inter', sans-serif !important;
    font-size     : 0.92rem !important;
    color         : var(--text-main) !important;
    transition    : border-color 0.18s, box-shadow 0.18s !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color : var(--blue-mid) !important;
    box-shadow   : 0 0 0 3px rgba(29,78,216,0.10) !important;
    background   : white !important;
    outline      : none !important;
}
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label {
    font-family   : 'Inter', sans-serif !important;
    font-size     : 0.75rem !important;
    font-weight   : 600 !important;
    color         : var(--text-muted) !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

div[data-testid="stButton"] > button {
    background    : var(--blue-dark) !important;
    color         : white !important;
    font-family   : 'Inter', sans-serif !important;
    font-weight   : 600 !important;
    font-size     : 0.88rem !important;
    border        : none !important;
    border-radius : 8px !important;
    padding       : 0.65rem 1.25rem !important;
    letter-spacing: 0.02em !important;
    transition    : background 0.2s, box-shadow 0.2s, transform 0.15s !important;
    box-shadow    : 0 2px 8px rgba(30,58,138,0.22) !important;
}
div[data-testid="stButton"] > button:hover {
    background : var(--blue-mid) !important;
    box-shadow : 0 4px 16px rgba(30,58,138,0.30) !important;
    transform  : translateY(-1px) !important;
}
div[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

.btn-secondary div[data-testid="stButton"] > button {
    background    : transparent !important;
    color         : var(--text-muted) !important;
    border        : 1.5px solid #e2e8f0 !important;
    box-shadow    : none !important;
    font-size     : 0.80rem !important;
    padding       : 0.38rem 0.9rem !important;
}
.btn-secondary div[data-testid="stButton"] > button:hover {
    border-color : var(--blue-dark) !important;
    color        : var(--blue-dark) !important;
    background   : var(--blue-light) !important;
    transform    : none !important;
    box-shadow   : none !important;
}

.card {
    background    : var(--card);
    border-radius : 14px;
    border        : 1px solid var(--border);
    box-shadow    : 0 1px 4px rgba(0,0,0,0.04), 0 4px 16px rgba(30,58,138,0.06);
    padding       : 1.75rem;
    margin-bottom : 1.25rem;
}

.navbar {
    background     : var(--card);
    border-radius  : 12px;
    border         : 1px solid var(--border);
    box-shadow     : 0 1px 4px rgba(0,0,0,0.04);
    padding        : 0.85rem 1.4rem;
    display        : flex;
    align-items    : center;
    justify-content: space-between;
    margin-bottom  : 1.5rem;
}

.badge {
    display       : inline-flex;
    align-items   : center;
    gap           : 0.35rem;
    background    : var(--blue-light);
    color         : var(--blue-dark);
    font-size     : 0.68rem;
    font-weight   : 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    padding       : 0.25rem 0.7rem;
    border-radius : 5px;
    margin-bottom : 1.2rem;
}

.result-card {
    background    : var(--card);
    border-radius : 14px;
    border        : 1px solid var(--border);
    box-shadow    : 0 4px 24px rgba(30,58,138,0.08);
    padding       : 2.5rem 1.75rem;
    text-align    : center;
    margin        : 1.25rem 0 1rem;
}
.result-icon {
    width         : 64px;
    height        : 64px;
    border-radius : 50%;
    display       : inline-flex;
    align-items   : center;
    justify-content: center;
    font-family   : 'Playfair Display', serif;
    font-size     : 1.8rem;
    font-weight   : 700;
    margin-bottom : 1.1rem;
}
.result-eyebrow {
    font-size     : 0.68rem;
    font-weight   : 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color         : var(--text-muted);
    margin-bottom : 0.4rem;
}
.result-status {
    font-family   : 'Playfair Display', 'Times New Roman', serif;
    font-size     : 2.1rem;
    font-weight   : 700;
    margin-bottom : 0.3rem;
}
.result-diag {
    font-size     : 0.88rem;
    color         : var(--text-muted);
    margin-bottom : 1.75rem;
}
.result-divider {
    width         : 32px;
    height        : 2px;
    background    : #e2e8f0;
    margin        : 0 auto 1.5rem;
    border-radius : 2px;
}
.result-score-label {
    font-size     : 0.68rem;
    font-weight   : 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color         : var(--text-muted);
    margin-bottom : 0.4rem;
}
.result-score {
    font-family   : 'Playfair Display', 'Times New Roman', serif;
    font-size     : 3.2rem;
    font-weight   : 600;
    letter-spacing: -0.03em;
    line-height   : 1;
}
.result-score-unit {
    font-family: 'Inter', sans-serif;
    font-size  : 1.2rem;
    font-weight: 500;
    color      : var(--text-muted);
}

.medical-alert {
    background   : var(--amber-bg);
    border       : 1px solid var(--amber-bdr);
    border-left  : 4px solid #f59e0b;
    border-radius: 8px;
    padding      : 0.9rem 1.1rem;
    font-size    : 0.84rem;
    color        : var(--amber-text);
    line-height  : 1.65;
    margin-bottom: 1rem;
}

div[data-testid="stTabs"] button {
    font-family   : 'Inter', sans-serif !important;
    font-size     : 0.85rem !important;
    font-weight   : 500 !important;
    color         : var(--text-muted) !important;
    border-bottom : 2px solid transparent !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color        : var(--blue-dark) !important;
    border-bottom: 2px solid var(--blue-dark) !important;
    font-weight  : 500 !important;
}

div[data-testid="stAlert"] { border-radius: 8px !important; }
details summary { font-size: 0.83rem !important; color: var(--text-muted) !important; }

.app-footer {
    text-align : center;
    color      : #94a3b8;
    font-size  : 0.74rem;
    padding    : 2rem 0 0.5rem;
    line-height: 1.9;
}

@media (max-width: 600px) {
    .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .result-score           { font-size: 2.6rem; }
    .result-status          { font-size: 1.7rem; }
}
</style>
""", unsafe_allow_html=True)


# UTILITAIRES — BASE DE DONNÉES
def get_conn():
    return psycopg2.connect(**DB_CONFIG)


# UTILITAIRES — AUTHENTIFICATION
def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    if not username or not password:
        return False, "Veuillez remplir tous les champs."
    if len(username) < 3:
        return False, "L'identifiant doit contenir au moins 3 caractères."
    if len(password) < 6:
        return False, "Le mot de passe doit contenir au moins 6 caractères."
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                (username, _hash(password))
            )
        return True, "Compte créé avec succès."
    except psycopg2.errors.UniqueViolation:
        return False, "Cet identifiant est déjà utilisé."
    except Exception as e:
        return False, f"Erreur base de données : {e}"

def login_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT password FROM users WHERE username = %s",
                (username,)
            )
            row = cur.fetchone()
        if row is None:
            return False, "Identifiant introuvable."
        if row[0] != _hash(password):
            return False, "Mot de passe incorrect."
        return True, "Connexion réussie."
    except Exception as e:
        return False, f"Erreur base de données : {e}"


# UTILITAIRES — DONNÉES
def validate_inputs(data: dict) -> bool:
    return all(v is not None and v > 0 for v in data.values())

def save_prediction(username: str, inputs: dict, prediction: str, probability: float) -> None:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions (
                    utilisateur,
                    radius_worst, texture_worst, perimeter_worst, area_worst,
                    smoothness_worst, compactness_worst, concavity_worst,
                    concave_points_worst, symmetry_worst, fractal_dimension_worst,
                    prediction, probability_pct
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                username,
                inputs["radius_worst"], inputs["texture_worst"],
                inputs["perimeter_worst"], inputs["area_worst"],
                inputs["smoothness_worst"], inputs["compactness_worst"],
                inputs["concavity_worst"], inputs["concave_points_worst"],
                inputs["symmetry_worst"], inputs["fractal_dimension_worst"],
                prediction, round(probability * 100, 2)
            ))
    except Exception as e:
        st.warning(f"Historique non enregistré : {e}")


# SESSION STATE
for key, default in [("authenticated", False), ("username", ""), ("auth_tab", "login")]:
    if key not in st.session_state:
        st.session_state[key] = default


# PAGE AUTHENTIFICATION
if not st.session_state.authenticated:

    st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align:center; margin-bottom:2rem;">
            <div style="
                width:70px; height:70px;
                background:var(--blue-dark);
                border-radius:14px;
                display:inline-flex; align-items:center; justify-content:center;
                font-size:1.7rem;
                box-shadow:0 6px 20px rgba(30,58,138,0.30);
                margin-bottom:1rem;">🩺</div>
            <div class="font-serif" style="
                font-size:1.85rem; font-weight:800;
                color:var(--text-main); letter-spacing:-0.02em;
                margin-bottom:0.25rem;">
                OncoScan AI
            </div>
            <div style="color:var(--text-muted); font-size:0.86rem;">
                Outil d'aide au diagnostic &nbsp;·&nbsp; Accès sécurisé
            </div>
        </div>
    """, unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["Se connecter", "Créer un compte"])

    with tab_login:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        login_user_input = st.text_input(
            "Identifiant", placeholder="Votre identifiant", key="login_user")
        login_pass_input = st.text_input(
            "Mot de passe", type="password", placeholder="••••••••", key="login_pass")
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        if st.button("Se connecter", use_container_width=True, key="btn_login"):
            ok, msg = login_user(login_user_input, login_pass_input)
            if ok:
                st.session_state.authenticated = True
                st.session_state.username = login_user_input.strip().lower()
                st.rerun()
            else:
                st.error(msg)

    with tab_register:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        
        reg_user_input  = st.text_input("Identifiant", placeholder="Choisissez un identifiant", key="reg_user")
        
        reg_pass_input  = st.text_input("Mot de passe", type="password", placeholder="6 caractères minimum", key="reg_pass")
        
        reg_pass2_input = st.text_input("Confirmer le mot de passe", type="password",
            placeholder="Confirmez le mot de passe", key="reg_pass2")
        
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        if st.button("Créer mon compte", use_container_width=True, key="btn_register"):
            if reg_pass_input != reg_pass2_input:
                st.error("Les mots de passe ne correspondent pas.")
            else:
                ok, msg = register_user(reg_user_input, reg_pass_input)
                if ok:
                    st.success(f"{msg} Vous pouvez maintenant vous connecter")
                else:
                    st.error(msg)

    st.markdown("""
        <div class="app-footer">
            🔒 Connexion sécurisée &nbsp;·&nbsp; Données confidentielles<br>
            OncoScan AI v1.0.0
        </div>
    """, unsafe_allow_html=True)

    st.stop()


# PAGE PRINCIPALE
st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

st.markdown(
    f"""<div class="navbar">
        <div class="font-serif" style="
            font-size:1.05rem; font-weight:700;
            color:var(--text-main); letter-spacing:-0.01em;
            display:flex; align-items:center; gap:0.5rem;">
            <span style="
                width:8px; height:8px;
                background:var(--blue-dark);
                border-radius:50%; display:inline-block;"></span>
            OncoScan AI
        </div>
        <div style="font-size:0.80rem; color:var(--text-muted);">
            <span style="color:var(--blue-dark); font-weight:600;">
                {st.session_state.username}
            </span>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

col_out, _ = st.columns([1.5, 5])

with col_out:
    st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
    if st.button("Déconnexion", key="btn_logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()
       
        
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

st.markdown("""
    <div style="margin-bottom:1.4rem;">
        <div class="font-serif" style="
            font-size:1.55rem; font-weight:800;
            color:var(--text-main); margin-bottom:0.35rem;">
            Breast Cancer Prediction
        </div>
    </div>
""", unsafe_allow_html=True)

# Formulaire
with st.container():
    inputs = {}
    col1, col2 = st.columns(2, gap="medium")
    for i, feature in enumerate(FEATURES):
        col = col1 if i % 2 == 0 else col2
        with col:
            inputs[feature] = st.number_input(
                label=FEATURE_LABELS[feature],
                min_value=0.0001,
                max_value=2300.0,
                value=None,
                step=0.0001,
                format="%.4f",
                placeholder="0.0000",
                key=feature,
            )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

if st.button("Lancer l'analyse", use_container_width=True, key="btn_predict"):

    if not validate_inputs(inputs):
        st.markdown("<p style='color:red; font-weight:bold;'>Veuillez renseigner tous les paramètres avant de lancer l'analyse.</p>",
                    unsafe_allow_html=True)
        st.stop()

    with st.spinner("Résultat en cours…"):
        try:
            response = requests.post(API_URL, json=inputs, timeout=TIMEOUT)

            if response.status_code == 200:
                result = response.json()

                # save_prediction (plus save_to_csv)
                save_prediction(
                    username   = st.session_state.username,
                    inputs     = inputs,
                    prediction = result["prediction"],
                    probability= result["probability"]
                )

                is_benign   = result["prediction"] == "B"
                badge_color = "var(--green)"    if is_benign else "var(--red)"
                bg_color    = "var(--green-bg)" if is_benign else "var(--red-bg)"
                icon = "✓"  if is_benign else "!"
                status_text = "NÉGATIF" if is_benign else "POSITIF"
                score = f"{result['probability'] * 100:.2f}"

                st.markdown(
                    '<div class="result-card">'
                      f'<div class="result-icon" style="background:{bg_color}; color:{badge_color};">'
                        f'{icon}'
                      '</div>'
                      '<div class="result-eyebrow">Résultat du test</div>'
                      f'<div class="result-status" style="color:{badge_color};">{status_text}</div>'
                      '<div class="result-divider"></div>'
                      '<div class="result-score-label">Score de confiance</div>'
                      f'<div class="result-score" style="color:{badge_color};">'
                        f'{score}<span class="result-score-unit"> %</span>'
                      '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.success("Merci pour votre confiance")

            elif response.status_code == 422:
                st.error("Données invalides envoyées à l'API")
                with st.expander("Détails"):
                    st.json(response.json())
            else:
                st.error(f"Erreur API — Code {response.status_code}")
                with st.expander("Détails"):
                    st.json(response.json())

        except Timeout:
            st.error("Délai dépassé — Vérifiez que le serveur API est démarré")
        except RequestException as e:
            st.error("Erreur de connexion à l'API.")
            with st.expander("Détails techniques"):
                st.code(str(e))
        except KeyError as e:
            st.error(f"Champ manquant dans la réponse API : {e}")
        except Exception as e:
            st.error("Erreur inattendue.")
            with st.expander("Détails"):
                st.code(f"{type(e).__name__}: {str(e)}")
                
    st.stop()

# Footer
st.markdown("""
    <div class="app-footer">
        🔒 Données confidentielles &nbsp;·&nbsp; OncoScan AI v1.0.0<br>
        FastAPI · Streamlit · PostgreSQL · Docker
    </div>
""", unsafe_allow_html=True)