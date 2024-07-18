import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from PIL import Image

#para ejecutar streamlit hay que copiar esto en el terminal
#streamlit run app.py

# Obtener el directorio actual y construir la ruta al archivo del modelo
directorio_actual = os.getcwd()
ruta_modelo = os.path.join(directorio_actual, 'model', 'best_rf_model.pkl')

# Verificar si el archivo existe
if not os.path.isfile(ruta_modelo):
    st.error(f"El archivo del modelo no se encuentra en la ruta: {ruta_modelo}")
else:
    # Cargar el modelo
    with open(ruta_modelo, 'rb') as file:
        model = joblib.load(file)


# Añadir CSS para cambiar el color de fondo y el estilo general
st.markdown(
    """
    <style>
    /* Fondo y estilo general */
    .main {
        background-color: #ffffff; /* Blanco */
        color: #000000; /* Negro para el texto predeterminado */
    }

    /* Botones */
    .stButton>button {
        color: white;
        background-color: #87CEEB; /* Azul cielo */
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }

    /* Títulos */
    h1 {
        color: #000080; /* Azul marino */
    }
    h2, h3, h4, h5, h6 {
        color: #87CEEB; /* Azul cielo */
    }
    /* Texto en los elementos de entrada y etiquetas */
    label, .stNumberInput label, .stSelectbox label, .stRadio label {
        color: #000000; /* Negro */
    }
    /* Texto de predicción */
    .stWarning, .stSuccess {
        color: #000000; /* Negro */
    }
    /* Texto del selectbox en la barra lateral */
    .sidebar .selectbox label {
        color: #ffffff; /* Blanco */
    }    
    </style>
    """, unsafe_allow_html=True
)

# Título y descripción
st.title("Predicción de Enfermedad Cardíaca")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning para predecir el riesgo de enfermedad cardíaca basado en diversos parámetros de salud.
""")

# Menú de navegación
menu = ["Explicación del Proyecto", "Predicción"]
choice = st.sidebar.selectbox("Menú", menu)

if choice == "Explicación del Proyecto":
    st.subheader("Explicación del Proyecto")
    st.markdown("""
    Este proyecto tiene como objetivo predecir el riesgo de enfermedad cardíaca utilizando diversos parámetros de salud del paciente.
    - **Edad**: Edad del paciente en años.
    - **Colesterol**: Nivel de colesterol en mg/dL.
    - **Frecuencia Cardíaca Máxima**: Máxima frecuencia cardíaca alcanzada durante el ejercicio.
    - **Depresión del ST**: Descenso del segmento ST inducido por el ejercicio.
    - **Presión Arterial en Reposo**: Presión arterial sistólica en reposo en mmHg.
    - **Tipo de Dolor de Pecho**: Tipos de dolor de pecho experimentados.
        - **ATA**: Angina Típica
        - **NAP**: Angina no Anginosa
        - **TA**: Angina Atípica
    - **Angina Inducida por Ejercicio**: Si el paciente experimenta angina inducida por el ejercicio.
    - **Azúcar en Sangre en Ayunas**: Si el nivel de azúcar en sangre en ayunas es alto.
    - **Electrocardiograma en Reposo**: Tipo de resultado del electrocardiograma en reposo.
        - **Normal**: Resultado electrocardiográfico normal.
        - **ST**: Hipertrofia ventricular izquierda detectada en el electrocardiograma.
    - **Pendiente del Segmento ST**: Pendiente del segmento ST durante el pico del ejercicio.
        - **Flat**: Pendiente plana del segmento ST.
        - **Up**: Pendiente ascendente del segmento ST.
    - **Género del Paciente**: Género del paciente (M: Masculino, F: Femenino).
        """)

    # Mostrar imágenes 
    image_path = os.path.join(directorio_actual, 'img') 
    images = ['corazon_inicio.jpg'] 
    for image_name in images:
        image = Image.open(os.path.join(image_path, image_name))
        st.image(image)

elif choice == "Predicción":
    st.subheader("Predicción del Riesgo de Enfermedad Cardíaca")

    # Crear formularios para ingresar datos del paciente
    st.markdown("### Información Básica del Paciente")
    age = st.number_input("Edad", 1, 120, help="Edad del paciente en años")
    cholesterol = st.number_input("Colesterol", 1, 500, help="Nivel de colesterol del paciente en mg/dL")
    maxhr = st.number_input("Frecuencia Cardíaca Máxima", 1, 220, help="Frecuencia cardíaca máxima alcanzada durante el ejercicio")
    oldpeak = st.number_input("Depresión del ST", 0.0, 10.0, help="Descenso del segmento ST inducido por el ejercicio")
    restingbp = st.number_input("Presión Arterial en Reposo", 1, 300, help="Presión arterial sistólica en reposo en mmHg")

    st.markdown("### Información Adicional del Paciente")
    chestpain_type = st.selectbox(
        "Tipo de Dolor de Pecho",
        ["ATA (Angina Típica)", "NAP (Angina no Anginosa)", "TA (Angina Atípica)"],
        help="Tipo de dolor de pecho experimentado"
        )
    exercise_angina = st.selectbox(
        "Angina Inducida por Ejercicio",
        ["No", "Sí"],
        help="Si el paciente experimenta angina inducida por el ejercicio"
        )
    fastingbs = st.selectbox(
        "Azúcar en Sangre en Ayunas",
        ["Normal", "Alto"],
        help="Si el nivel de azúcar en sangre en ayunas es alto"
        )
    restingecg = st.selectbox(
        "Electrocardiograma en Reposo",
        ["Normal", "ST"],
        help="Tipo de resultado del electrocardiograma en reposo"
        )
    st_slope = st.selectbox(
        "Pendiente del Segmento ST",
        ["Up (Ascendente)", "Flat (Plana)"],
        help="Pendiente del segmento ST durante el pico del ejercicio"
        )
    sex = st.selectbox(
        "Género del Paciente",
        ["M (Masculino)", "F (Femenino)"],
        help="Género del paciente"
        )

    # Codificar variables categóricas
    chestpain_ata = 1 if chestpain_type == "ATA (Angina Típica)" else 0
    chestpain_nap = 1 if chestpain_type == "NAP (Angina no Anginosa)" else 0
    chestpain_ta = 1 if chestpain_type == "TA (Angina Atípica)" else 0
    exercise_angina_y = 1 if exercise_angina == "Sí" else 0
    fastingbs_true = 1 if fastingbs == "Alto" else 0
    restingecg_normal = 1 if restingecg == "Normal" else 0
    restingecg_st = 1 if restingecg == "ST" else 0
    st_slope_flat = 1 if st_slope == "Flat (Plana)" else 0
    st_slope_up = 1 if st_slope == "Up (Ascendente)" else 0
    sex_m = 1 if sex == "M (Masculino)" else 0

    # Crear un dataframe con los datos del paciente
    patient_data = pd.DataFrame({
        'Age': [age],
        'Cholesterol': [cholesterol],
        'MaxHR': [maxhr],
        'Oldpeak': [oldpeak],
        'RestingBP': [restingbp],
        'ChestPainType_ATA': [chestpain_ata],
        'ChestPainType_NAP': [chestpain_nap],
        'ChestPainType_TA': [chestpain_ta],
        'ExerciseAngina_Y': [exercise_angina_y],
        'FastingBS_True': [fastingbs_true],
        'RestingECG_Normal': [restingecg_normal],
        'RestingECG_ST': [restingecg_st],
        'ST_Slope_Flat': [st_slope_flat],
        'ST_Slope_Up': [st_slope_up],
        'Sex_M': [sex_m]
    })

    # Normalizar características numéricas
    numerical_cols = ['Age', 'Cholesterol', 'MaxHR', 'Oldpeak', 'RestingBP']
    scaler = StandardScaler()
    patient_data[numerical_cols] = scaler.fit_transform(patient_data[numerical_cols])

    if st.button("Predecir"):
        prediction = model.predict(patient_data)
        prediction_proba = model.predict_proba(patient_data)

        if prediction[0] == 1:
            st.warning(f"El paciente tiene un riesgo alto de enfermedad cardíaca con una probabilidad de {prediction_proba[0][1]* 100:.2f} %")
        else:
            st.success(f"El paciente tiene un riesgo bajo de enfermedad cardíaca con una probabilidad de {prediction_proba[0][0]* 100:.2f} %")
    