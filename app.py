import streamlit as st
import pandas as pd #type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import json
from dotenv import dotenv_values
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams


#wczytanie danych
@st.cache_data
def load_data():
    df = pd.read_csv("welcome_survey_simple_v2.csv", sep=";")
    df.fillna("Brak", inplace=True)
    # Tworzenie kolumny 'combined'
    df['combined'] = df.apply(lambda row: f"{row['fav_animals']} {row['fav_place']} {row['gender']} {row['edu_level']}", axis=1)
    return df

df = pd.read_csv("welcome_survey_simple_v2.csv", sep=";")

# Inicjalizacja Qdrant
qdrant_client =QdrantClient(location=":memory:")


#--kolor tła--
st.markdown(
    """
    <style>
    /* Tło głównej części aplikacji (granatowe) i biały tekst */
    .stApp {
        background-color: #1B263B;
        color: white;
    }
    /* Tło paska bocznego (jasnoróżowe) i granatowe napisy */
    .stSidebar {
        background-color: #ffb6c1;
        color: #1B263B;
    }
    /* Granatowy kolor napisów w elementach paska bocznego */
    .stSidebar * {
        color: #1B263B;
    }
    /* Jasnoróżowe tło dla st.multiselect */
    .stMultiSelect [data-baseweb="select"] {
        background-color: #ffb6c1;
        border-radius: 5px;
    }
    /* Granatowy kolor tekstu w opcjach multiselect */
    .stMultiSelect [data-baseweb="select"] * {
        color: #1B263B;
    /* Zmiana koloru tekstu w elementach DataFrame */
    .dataframe {
        color: #FF69B4; /* Różowy */
    }
    </style>
    """,
    unsafe_allow_html=True
)
env = dotenv_values(".env")

EMBEDDING_DIM=1536
EMBEDDIING_MODEL="text-embedding-3-small"
QDRANT_COLLECTION_NAME = "welcome_survey"
qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION_NAME)
MODEL_NAME='welcome_survey_clustering_pipeline_v2'
DATA= 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

def get_openai_client():
    return OpenAI(api_key=env["OPENAI_API_KEY"])


#
# DB
#


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path=":memory:")
    url=env["QDRANT_URL"], 
    api_key=env["QDRANT_API_KEY"],
    
# Funkcje Qdrant

def assure_db_collection_exists(qdrant_client,collection_name,embedding_dim):
    if not qdrant_client.collection_exists(collection_name):
        print(f"Tworze kolekcje'{collection_name}'")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )
    else:
        print(f"Kolekcja '{collection_name}' juz istnieje")
        
def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDIING_MODEL,
        dimensions=EMBEDDING_DIM,
    )

    return result.data[0].embedding

def list_notes_from_db(
    qdrant_client,
    collection_name,
    query=None,
):
    if not query:
        notes=qdrant_client.search(collection_name=QDRANT_COLLECTION_NAME,query_vector=get_embedding(query),limit=20)[0]
        result=[]
        for note in notes:
            result.append({
                "text":note.payload["text"],
                "score":None,   
            })
            
        return result
    else:
        notes = qdrant_client.search(
            collection_name=collection_name,
            query_vector=get_embedding(text=query),
            limit=10,
        )
        result = []
        for note in notes:
            result.append({
                "text": note.payload["text"],
                "score": note.score,
            })

        return result




# Wczytaj dane i wstaw do Qdrant

def add_note_to_db(note_text):
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        exact=True,
    )
    qdrant_client.upsert(
    collection_name=QDRANT_COLLECTION_NAME,
    points=[
        PointStruct(
            id=idx,
            vector=get_embedding(row ["combined"]),
            payload={
                "age": row["age"],
                "edu_level": row["edu_level"],
                "fav_animals": row["fav_animals"],
                "fav_place": row["fav_place"],
                "gender": row["gender"],
            },
        )
        for idx, row in df.iterrows()
    ]
)
       
@st.cache_data
def get_model():
    return load_model(MODEL_NAME)


@st.cache_data
def get_all_participants():
    model=get_model()
    all_df=pd.read_csv(DATA, sep=';')
    df_with_clusters=predict_model(model, data=all_df)
    return df_with_clusters

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())
    

# Interfejs Streamlit    
st.title("Znajdz znajomych")

with st.sidebar:
    st.header("Opowiedz mi coś o sobie! 🗣️")
    st.markdown("Cześć! 👋 Jestem tu, aby pomóc Ci znaleźć osoby o podobnych zainteresowaniach! 🤩")
    
        # Styl dla całego sidebaru z różową czcionką
    st.markdown("""
    <style>
    /* Styl dla całego sidebaru */
    .css-1d391kg, .css-1d391kg > div, .css-1d391kg > div > div {
        border: 2px solid #1E3A8A !important;
        border-radius: 10px !important;
        padding: 15px !important;
        background-color: #EFF6FF !important;
    }
    
    /* Jasnoróżowy kolor czcionki dla wszystkich tekstów w sidebarze */
    .css-1d391kg, 
    .css-1d391kg h1, 
    .css-1d391kg h2, 
    .css-1d391kg h3,
    .css-1d391kg p,
    .css-1d391kg label,
    .css-1d391kg div,
    .stSelectbox label,
    .stRadio label,
    .stMarkdown {
        color: #FFB6C1 !important;
        font-weight: bold !important;
    }
    
    /* Styl dla selectbox i radio */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stRadio"] > div {
        border: 2px solid #FFB6C1 !important;
        border-radius: 8px !important;
        padding: 8px !important;
        background-color: rgba(255, 182, 193, 0.1) !important;
    }
    
    /* Styl dla opcji w dropdown */
    .st-bd, .st-bc, .st-bb {
        background-color: rgba(255, 182, 193, 0.1) !important;
        color: #FFB6C1 !important;
    }
    
    /* Styl dla hover na opcjach */
    .st-bd:hover, .st-bc:hover, .st-bb:hover {
        background-color: #FFB6C1 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
   
    age = st.selectbox("Wiek 🎂", [
        '<18 👶', 
        '18-24 🎓', 
        '25-34 💼', 
        '35-44 🏡', 
        '45-54 👨‍💼', 
        '55-64 👴', 
        '>=65 🧓', 
        'unknown ❓'
    ])
    
    edu_level = st.selectbox("Wykształcenie 🎓", [
        'Podstawowe 📚', 
        'Średnie 🎒', 
        'Wyższe 🎓'
    ])
    fav_animals = st.selectbox("Ulubione zwierzęta 🐾", ['Brak ulubionych', 'Psy 🐶', 'Koty 🐱', 'Inne 🦜', 'Koty i Psy 🐕🐈'])
    fav_place = st.selectbox("Ulubione miejsce 🌍", ['Nad wodą 🌊', 'W lesie 🌲', 'W górach 🏔️', 'Inne ❓'])
    gender = st.radio("Płeć 👤", ['Mężczyzna 👨', 'Kobieta 👩'])

person_df = pd.DataFrame([
    {
        'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
    }    
])     
model=get_model()
all_df=get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id= predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data=cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Najbliżej Ci do klastra {predicted_cluster_id}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)
