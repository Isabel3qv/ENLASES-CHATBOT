# streamlit_app.py
# Interfaz Web Real para el Asistente RAG de Enlaces El Salvador

import os
import numpy as np
import streamlit as st
import time # Para simular el tiempo de respuesta del chatbot

# Importaciones del motor RAG (Necesitas que estas librerías estén instaladas)
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
from google import genai
from google.genai.errors import APIError

# ==============================================================================
# 1. CORPUS DE CONOCIMIENTO EXPANDIDO (Base de Datos)
# ==============================================================================
data_enlaces_sv_expanded = """
**Soporte Oficial de Enlaces El Salvador (MINED)**

1.  **Obtención y Formato del Correo Electrónico:**
    * **Estudiantes:** El formato es [primer_nombre].[primer_apellido]@[clases.edu.sv]. Se consulta usando el **NIE** y fecha de nacimiento en el **Portal Estudiantil Oficial**.
    * **Docentes/Administrativos:** Usan el dominio @docente.edu.sv.

2.  **Contraseña Inicial y Restablecimiento:**
    * **Contraseña Inicial (Estudiante):** Suele ser el NIE completo seguido del símbolo asterisco y los dos últimos dígitos del año de nacimiento (Ej: NIE*YY).
    * **Restablecimiento por Olvido:** Si olvidó su contraseña, debe solicitar el restablecimiento contactando al **Coordinador de Enlaces** de su centro educativo o enviando un correo al área de soporte.

3.  **Reporte de Robo o Pérdida de Equipo (Tabletas/Laptops):**
    * En caso de robo o pérdida, debe **reportar inmediatamente** a la dirección de su centro educativo para iniciar la gestión de la reposición.

4.  **Enlaces y Contactos de Soporte Técnico:**
    * **Soporte Telefónico General (MINED):** Llame al **2592-7000**. Horario: Lunes a Viernes, 8:00 AM a 4:00 PM.
    * **Correo de Soporte Técnico:** Envíe consultas a soporte_tecnico@mined.edu.sv.
    * **Portal Oficial Enlaces:** URL para información y servicios: www.enlaces.edu.sv.
"""
K_VALUE = 3 # Constante de búsqueda RAG

# ==============================================================================
# 2. FUNCIONES DEL RAG (Configuración y Lógica)
# ==============================================================================

@st.cache_resource
def setup_rag():
    """Configura el cliente de Gemini y la base de datos vectorial FAISS una sola vez."""
    try:
        gemini_api_key = os.environ.get("GEMINI_API_KEY") 
        if not gemini_api_key:
            st.error("🚨 Error: La clave GEMINI_API_KEY no está configurada. Ejecuta Streamlit con la clave API.")
            return None, None, None, None, False 

        client = genai.Client(api_key=gemini_api_key)
        
        # Inicialización de modelos de embeddings
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Creación de chunks y embeddings
        chunks = [data_enlaces_sv_expanded[i:i + 500] for i in range(0, len(data_enlaces_sv_expanded), 500)]
        embeddings = embedding_model.encode(chunks, show_progress_bar=False)
        
        # Creación del índice FAISS
        dimension = embeddings.shape[1]
        faiss_index = IndexFlatL2(dimension)
        faiss_index.add(np.array(embeddings))
        
        return client, embedding_model, faiss_index, chunks, True

    except Exception as e:
        st.error(f"⚠️ Error al inicializar el RAG: {e}")
        # CORRECCIÓN DE VALOR: Devolvemos 5 valores para evitar ValueError
        return None, None, None, None, False

# Cargar el RAG y almacenar variables en caché (¡Esta línea es vital!)
client, embedding_model, faiss_index, knowledge_chunks, llm_activo = setup_rag()


def retrieve_context(query):
    """Busca los K_VALUE (3) fragmentos de texto más relevantes."""
    query_embedding = embedding_model.encode([query])
    D, I = faiss_index.search(np.array(query_embedding), K_VALUE)
    return [knowledge_chunks[i] for i in I[0]]

def generate_rag_answer(context, query):
    """Genera la respuesta usando Gemini con el tono amigable."""
    if not llm_activo: return "El motor de IA no está activo. Por favor, revisa la configuración de la clave API.", context
    
    # 🎯 PROMPT DE ROL PARA GEMINI (CONCISO Y ÚTIL)
    prompt = f"""
    Eres un asistente amigable y profesional de Soporte Enlaces El Salvador.
    Tu tarea principal es proporcionar información de soporte **directamente** de la sección CONTEXTO.
    
    Regla 1: Sé conciso. Para preguntas simples (como 'hola'), responde con un saludo breve y pregunta cómo puedes ayudar.
    Regla 2: Para preguntas sobre soporte, usa un tono accesible (no técnico) y solo la información del CONTEXTO.
    Regla 3: Si el CONTEXTO no contiene la respuesta, di amablemente que la información específica no está disponible en la base de datos de soporte.

    CONTEXTO:
    ---
    {'\n'.join(context)}
    ---

    PREGUNTA DEL USUARIO: {query}

    RESPUESTA:
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=prompt,
        config=genai.types.GenerateContentConfig(temperature=0.5)
    )
    # Devolvemos el texto generado y el contexto (fuentes)
    return response.text, context

# ==============================================================================
# 3. INTERFAZ WEB CON STREAMLIT (Estilizado Web y Lógica)
# ==============================================================================

# 🎨 ESTILIZADO WEB: Fuerza color negro, fuente grande y diseño
st.set_page_config(page_title="Soporte Enlaces SV", page_icon="⭐", layout="wide") 

st.markdown("""
<style>
/* 1. CONFIGURACIÓN GENERAL, FUENTE GRANDE Y FONDO */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; 
    font-size: 1.1em;
}
.stApp {
    background-color: #f0ffef; /* Fondo verde muy suave */
}

/* 2. ARREGLO DE VISIBILIDAD DE CHAT (FORZAR TEXTO NEGRO) */

/* Fuerza el color del texto a negro para CUALQUIER ELEMENTO DENTRO DEL CHAT */
div[data-testid="stChatMessage"] * {
    color: black !important;
}

/* Estilo de los mensajes del asistente (fondo suave) */
div[data-testid="stChatMessage"][data-index$="1"] {
    background-color: #f0f0f0 !important; /* Gris claro para el asistente */
    border-radius: 15px;
}

/* Estilo de los mensajes del usuario (fondo suave) */
div[data-testid="stChatMessage"][data-index$="0"] {
    background-color: #d8eaff !important; /* Azul muy claro para el usuario */
    border-radius: 15px;
}

/* ESTILIZADO ADICIONAL DE AVATARES (FONDO GRIS) */
/* Aplicar fondo gris a los avatares (iconos) */
.stChatMessage div[data-testid="stChatMessageAvatar"] {
    background-color: #e0e0e0; /* Fondo gris para el avatar */
    border-radius: 50%; /* Asegura que sea un círculo */
    padding: 5px; /* Pequeño padding alrededor del icono */
    line-height: 1; /* Para centrar bien el texto del icono */
}

/* 3. ESTILO DEL TÍTULO Y CONTENEDOR DE LOGOS (AZUL INSTITUCIONAL) */
.title-container {
    display: flex;
    align-items: center;
    padding: 20px 10px;
    background-color: #0d6efd;
    color: white;
    border-radius: 8px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
}
.title-container h1 {
    color: white !important;
    margin: 0;
    font-size: 1.7em;
    padding-left: 15px;
}

/* 4. TAMAÑO DEL TEXTO DEL CHAT: Aumentamos un poco más la fuente de los mensajes */
.stChatMessage {
    font-size: 1.15em !important; 
}
</style>
""", unsafe_allow_html=True)

# Encabezado con logos y estilizado
st.markdown(f"""
<div class="title-container">
    <span style='font-size: 50px;'>🎓</span>
    <span style='font-size: 50px; margin-right: 15px;'>💡</span>
    <h1>Asistente Virtual de Soporte - Enlaces MINED</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("<p style='color: #444444; font-size: 1.05em;'>Pregunta sobre correos, contraseñas, soporte o reportes de robo. Usamos una base de conocimiento oficial para darte respuestas precisas.</p>", unsafe_allow_html=True)

# Inicializar historial de chat
if "messages" not in st.session_state:
    # 🌟 CAMBIO 1: Añadimos avatar="🤖"
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu asistente amigable de soporte Enlaces. ¿En qué te puedo ayudar hoy?"}]

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    # 🌟 CAMBIO 2: Lógica para determinar el avatar a mostrar
    avatar_icon = "🤖" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# Captura la entrada del usuario y ejecuta la lógica
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    # 1. Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 🌟 CAMBIO 3: Añadimos avatar="👤"
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Generar respuesta del Asistente
    # 🌟 CAMBIO 4: Añadimos avatar="🤖"
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤖 Buscando contexto y generando respuesta..."):
            
            # --- Lógica RAG ---
            contexto_recuperado = retrieve_context(prompt)
            respuesta, fuentes = generate_rag_answer(contexto_recuperado, prompt)
            
            # 1. Mostrar la respuesta limpia del Bot en el chat
            st.markdown(respuesta.strip()) 
            
            # 2. Crear un Expander para las fuentes (se muestran aparte)
            with st.expander("🔍 Ver Contexto Utilizado (Fuentes)"):
                st.markdown("**Fragmentos de la base de datos que guiaron esta respuesta:**")
                
                # Iterar y mostrar cada fragmento en un bloque limpio
                for i, fragmento in enumerate(contexto_recuperado):
                    fragmento_limpio = fragmento.replace('**', '').replace('\n*', ' ').strip()
                    st.code(f"Fragmento {i+1}:\n{fragmento_limpio}", language="markdown")
            
            # 3. Guardamos solo la respuesta limpia en el historial (sin fuentes)
            final_response = respuesta.strip() 
    
    # 3. Agregar la respuesta al historial
    st.session_state.messages.append({"role": "assistant", "content": final_response})