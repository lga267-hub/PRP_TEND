import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st

# --- ENTRENAMIENTO DEL MODELO (se ejecuta una sola vez al iniciar la app) ---
# Carga los datos existentes para entrenar el modelo
df = pd.read_excel("datos_pacientes.xlsx")
X_completo = df[['edad', 'sexo', 'dolor_nocturno, tipo_rotura, fm']]
y_completo = df['exito']

# Crea y entrena el modelo definitivo
modelo_final = LogisticRegression()
modelo_final.fit(X_completo, y_completo)

# --- INTERFAZ DE USUARIO DE LA APLICACIÓN WEB ---

# Título de la aplicación
st.title('⚕️ Predictor de Respuesta a PRP en Tendinopatía')

# Barra lateral para introducir los datos del paciente
st.sidebar.header('Datos del Paciente')

def obtener_datos_paciente():
    """Función para recoger los datos introducidos por el usuario."""
    edad = st.sidebar.slider('Edad del paciente', 20, 90, 50)
    
    sexo_texto = st.sidebar.selectbox('Sexo', ('Hombre', 'Mujer'))
    sexo_num = 0 if sexo_texto == 'Hombre' else 1
    
    dolor_texto = st.sidebar.selectbox('¿Presenta dolor nocturno?', ('No', 'Sí'))
    dolor_num = 0 if dolor_texto == 'No' else 1

    fm_texto = st.sidebar.selectbox('¿Presenta antecedentes de dolor nociplástico?', ('No', 'Sí'))
    fm_num = 0 if dolor_texto == 'No' else 1

    tipo_rotura_texto = st.sidebar.slider('Tipo de rotura', ('Parcial', 'Espesor completo'))
    tipo_num = 0 if tipo_rotura_texto == 'Espesor completo' else 1
   
    # Crea un DataFrame con los datos para el modelo
    datos_paciente = pd.DataFrame({
        'edad': [edad],
        'sexo': [sexo_num],
        'dolor_nocturno': [dolor_num],
        'fm': [fm_num],
        'tipo_rotura': [tipo_num]
                    
    })
    return datos_paciente

# Guardamos los datos introducidos
df_paciente = obtener_datos_paciente()

# --- REALIZAR LA PREDICCIÓN Y MOSTRAR RESULTADOS ---

# Botón para ejecutar la predicción
if st.sidebar.button('Predecir Probabilidad de Éxito'):
    
    # Realizar la predicción
    probabilidades = modelo_final.predict_proba(df_paciente)
    prob_de_exito = probabilidades[0][1]
    
    st.subheader('Resultado de la Predicción')
    st.write(f'La probabilidad de éxito del tratamiento para este paciente es de:')
    
    # Mostrar el resultado con formato
    st.markdown(f'<p style="font-size:24px; color:{"green" if prob_de_exito > 0.5 else "red"};">{prob_de_exito:.2%}</p>', unsafe_allow_html=True)
    
    # Barra de progreso visual
    st.progress(prob_de_exito)

else:
    st.info('Por favor, introduce los datos del paciente en la barra lateral y haz clic en "Predecir".')