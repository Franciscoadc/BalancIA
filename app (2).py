
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.title("BalancIA - Tutor de Destilación")
st.write("Haz tus preguntas sobre destilación, balances de materia y energía.")

@st.cache_resource
def cargar_modelo():
    modelo = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    model = AutoModelForCausalLM.from_pretrained(modelo, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

tokenizer, model = cargar_modelo()

prompt_inicial = '''
Eres BalancIA, un tutor experto en procesos de destilación y balances de materia y energía. Tu misión es enseñar y ayudar a estudiantes de ingeniería química.
Responde con explicaciones claras, ejemplos cuando sea necesario, y pasos detallados. Siempre verifica que el estudiante entienda los conceptos de:
- Tipos de destilación
- Diseño de columnas
- Balances de materia y energía
- Cálculo de número de platos teóricos
- Parámetros operativos: reflujo, presión, temperatura.
Si el estudiante comete un error o tiene confusión, explícale pacientemente y sugiere cómo mejorar su razonamiento.

Pregunta: {pregunta}
Respuesta:
'''

pregunta = st.text_area("Escribe tu pregunta aquí:")

if st.button("Preguntar a BalancIA"):
    if pregunta.strip() == "":
        st.warning("Por favor escribe una pregunta.")
    else:
        prompt = prompt_inicial.format(pregunta=pregunta)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=300)
        respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
        respuesta_final = respuesta.split("Respuesta:")[-1].strip()
        st.success(respuesta_final)
