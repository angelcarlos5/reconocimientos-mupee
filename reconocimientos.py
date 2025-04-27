
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer('all-MiniLM-L6-v2')

modelo = cargar_modelo()

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("reconocimientos_total_limpio.csv")
    return df

df = cargar_datos()

# Crear embeddings para máster, universidad y asignatura cursada
@st.cache_data
def vectorizar_campos(df):
    embeddings_master = modelo.encode(df["MÁSTER CURSADO"].tolist(), convert_to_tensor=True)
    embeddings_universidad = modelo.encode(df["UNIVERSIDAD DE PROCEDENCIA"].tolist(), convert_to_tensor=True)
    embeddings_asignatura = modelo.encode(df["ASIGNATURA CURSADA"].tolist(), convert_to_tensor=True)
    return embeddings_master, embeddings_universidad, embeddings_asignatura

embeddings_master, embeddings_universidad, embeddings_asignatura = vectorizar_campos(df)

# App
st.title("🎓 Asistente Inteligente de Reconocimiento MUPEE - VERSIÓN MEJORADA")

tab1, tab2 = st.tabs(["🔎 Buscar reconocimiento", "🆕 Añadir nuevo reconocimiento"])

with tab1:
    st.subheader("Introduce los datos del reconocimiento solicitado:")

    master_cursado = st.text_input("Máster cursado (puedes escribir parcialmente)")
    universidad_origen = st.text_input("Universidad de procedencia (puedes escribir parcialmente)")
    anio_academico = st.text_input("Año académico del máster cursado (opcional)")
    asignatura_aportada = st.text_area("Asignatura que aporta el alumno")

    if st.button("🔎 Buscar reconocimiento"):
        if asignatura_aportada:
            embed_master_input = modelo.encode(master_cursado.upper(), convert_to_tensor=True)
            embed_universidad_input = modelo.encode(universidad_origen.upper(), convert_to_tensor=True)
            embed_asignatura_input = modelo.encode(asignatura_aportada.upper(), convert_to_tensor=True)

            # Calcular similitudes
            df["SIM_MASTER"] = util.cos_sim(embed_master_input, embeddings_master)[0].cpu().numpy()
            df["SIM_UNIVERSIDAD"] = util.cos_sim(embed_universidad_input, embeddings_universidad)[0].cpu().numpy()
            df["SIM_ASIGNATURA"] = util.cos_sim(embed_asignatura_input, embeddings_asignatura)[0].cpu().numpy()

            # Contar campos que cumplen similitud > 0.5
            df["COINCIDENCIAS"] = (
                (df["SIM_MASTER"] > 0.5).astype(int) +
                (df["SIM_UNIVERSIDAD"] > 0.5).astype(int) +
                (df["SIM_ASIGNATURA"] > 0.5).astype(int)
            )

            # Filtrar registros que tienen al menos 2 coincidencias buenas
            df_filtrado = df[df["COINCIDENCIAS"] >= 2]

            if anio_academico:
                df_filtrado = df_filtrado[df_filtrado["AÑO ACADÉMICO"] == anio_academico.upper()]

            resultados = df_filtrado.sort_values(["SIM_MASTER", "SIM_UNIVERSIDAD", "SIM_ASIGNATURA"], ascending=False)

            if not resultados.empty:
                st.success(f"🎯 Casos similares encontrados: {len(resultados)}")
                st.dataframe(resultados[[
                    "ASIGNATURA CURSADA", "ASIGNATURA RECONOCIDA EN MUPEE", 
                    "MÁSTER CURSADO", "UNIVERSIDAD DE PROCEDENCIA", "AÑO ACADÉMICO", 
                    "SIM_MASTER", "SIM_UNIVERSIDAD", "SIM_ASIGNATURA"
                ]].reset_index(drop=True))

                # Estadísticas
                asignaturas_reconocidas = resultados["ASIGNATURA RECONOCIDA EN MUPEE"].nunique()
                porcentaje = (asignaturas_reconocidas / len(resultados)) * 100
                st.markdown(f"**📊 Porcentaje de asignaturas reconocidas respecto a coincidencias:** `{porcentaje:.1f}%`")
            else:
                st.warning("❗ No se encontraron coincidencias suficientes con los datos aportados. Intenta ajustar los términos.")
        else:
            st.error("Por favor, escribe al menos la asignatura aportada.")

with tab2:
    st.subheader("🆕 Registrar un nuevo reconocimiento")
    nuevo_master = st.text_input("Nuevo máster cursado (origen)")
    nueva_universidad = st.text_input("Nueva universidad de procedencia")
    nuevo_anio = st.text_input("Nuevo año académico")
    nueva_asignatura_aportada = st.text_area("Nueva asignatura aportada")
    nueva_asignatura_reconocida = st.text_input("Asignatura que se reconoce en MUPEE")

    if st.button("💾 Guardar nuevo reconocimiento"):
        if (nuevo_master and nueva_universidad and nuevo_anio and 
            nueva_asignatura_aportada and nueva_asignatura_reconocida):

            nuevo_registro = pd.DataFrame({
                "MÁSTER CURSADO": [nuevo_master.upper()],
                "AÑO ACADÉMICO": [nuevo_anio.upper()],
                "UNIVERSIDAD DE PROCEDENCIA": [nueva_universidad.upper()],
                "ASIGNATURA CURSADA": [nueva_asignatura_aportada.upper()],
                "ASIGNATURA RECONOCIDA EN MUPEE": [nueva_asignatura_reconocida.upper()]
            })

            df_actual = pd.read_csv("reconocimientos_total_limpio.csv")
            df_actual = pd.concat([df_actual, nuevo_registro], ignore_index=True)
            df_actual.to_csv("reconocimientos_total_limpio.csv", index=False)

            st.success("✅ Nuevo reconocimiento registrado correctamente.")

    st.divider()
    st.download_button("📥 Descargar base de datos actualizada", 
                       data=pd.read_csv("reconocimientos_total_limpio.csv").to_csv(index=False),
                       file_name="reconocimientos_actualizado.csv",
                       mime="text/csv")
