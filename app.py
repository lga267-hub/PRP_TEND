import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st

# --- ENTRENAMIENTO DEL MODELO ---

# 1. Lee tu archivo .csv y le dice que use punto y coma (;)
df = pd.read_csv("datos_pacientes.csv", sep=';') 

# 2. Limpia espacios en blanco de los nombres de columnas (por seguridad)
df.columns = df.columns.str.strip()

# 3. Define la lista de variables CORRECTA (basada en tu CSV)
X_completo = df[['edad', 'sexo', 'fm', 'tipo_rotura', 'dolor_nocturno']]
y_completo = df['exito']

# 4. Crea y entrena el modelo definitivo
modelo_final = LogisticRegression()
modelo_final.fit(X_completo, y_completo)

# --- INTERFAZ DE USUARIO DE LA APLICACIÓN WEB ---

st.title('⚕️ Predictor de Respuesta a PRP en Tendinopatía')
st.sidebar.header('Datos del Paciente')

def obtener_datos_paciente():
    """Función para recoger los datos introducidos por el usuario."""
    
    # Control para Edad
    edad = st.sidebar.slider('Edad del paciente', 20, 90, 50)
    
    # Control para Sexo
    sexo_texto = st.sidebar.selectbox('Sexo', ('Hombre', 'Mujer'))
    sexo_num = 0 if sexo_texto == 'Hombre' else 1
    
    # Control para Dolor Nocturno
    dolor_texto = st.sidebar.selectbox('¿Presenta dolor nocturno?', ('No', 'Sí'))
    dolor_num = 0 if dolor_texto == 'No' else 1

    # Control para Dolor Nociplástico (fm)
    fm_texto = st.sidebar.selectbox('¿Presenta antecedentes de dolor nociplástico?', ('No', 'Sí'))
    fm_num = 0 if fm_texto == 'No' else 1 

    # Control para Tipo de Rotura (¡Corregido a selectbox!)
    tipo_rotura_texto = st.sidebar.selectbox('Tipo de rotura', ('Parcial', 'Espesor completo'))
    # Asignamos 0 a 'Espesor completo' y 1 a 'Parcial' (como en tu CSV)
    tipo_num = 0 if tipo_rotura_texto == 'Espesor completo' else 1
   
    # 5. Crea el DataFrame para el nuevo paciente (¡nombres correctos!)
    datos_paciente = pd.DataFrame({
        'edad': [edad],
        'sexo': [sexo_num],
        'fm': [fm_num],
        'tipo_rotura': [tipo_num],
        'dolor_nocturno': [dolor_num]
    })
    return datos_paciente

# Guardamos los datos introducidos
df_paciente = obtener_datos_paciente()

# --- REALIZAR LA PREDICCIÓN Y MOSTRAR RESULTADOS ---

if st.sidebar.button('Predecir Probabilidad de Éxito'):
    
    probabilidades = modelo_final.predict_proba(df_paciente)
    prob_de_exito = probabilidades[0][1]
    
    st.subheader('Resultado de la Predicción')
    st.write(f'La probabilidad de éxito del tratamiento para este paciente es de:')
    
    st.markdown(f'<p style="font-size:24px; color:{"green" if prob_de_exito > 0.5 else "red"};">{prob_de_exito:.2%}</p>', unsafe_allow_html=True)
    st.progress(prob_de_exito)

else:
    st.info('Por favor, introduce los datos del paciente en la barra lateral y haz clic en "Predecir".')