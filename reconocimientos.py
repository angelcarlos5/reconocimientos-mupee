
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("reconocimientos_total_limpio.csv")
    return df

df = cargar_datos()

# Función de búsqueda usando TF-IDF
def buscar_similitud(campo_usuario, columna_datos):
    vectorizer = TfidfVectorizer().fit_transform([campo_usuario] + columna_datos.tolist())
    similitudes = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    return similitudes

# App
st.title("🎓 Asistente Inteligente de Reconocimiento MUPEE - VERSIÓN LIGERA")

tab1, tab2 = st.tabs(["🔎 Buscar reconocimiento", "🆕 Añadir nuevo reconocimiento"])

with tab1:
    st.subheader("Introduce los datos del reconocimiento solicitado:")

    master_cursado = st.text_input("Máster cursado (puedes escribir parcialmente)")
    universidad_origen = st.text_input("Universidad de procedencia (puedes escribir parcialmente)")
    anio_academico = st.text_input("Año académico del máster cursado (opcional)")
    asignatura_aportada = st.text_area("Asignatura que aporta el alumno")

    if st.button("🔎 Buscar reconocimiento"):
        if asignatura_aportada:
            sim_master = buscar_similitud(master_cursado.upper(), df["MÁSTER CURSADO"])
            sim_universidad = buscar_similitud(universidad_origen.upper(), df["UNIVERSIDAD DE PROCEDENCIA"])
            sim_asignatura = buscar_similitud(asignatura_aportada.upper(), df["ASIGNATURA CURSADA"])

            df["SIM_MASTER"] = sim_master
            df["SIM_UNIVERSIDAD"] = sim_universidad
            df["SIM_ASIGNATURA"] = sim_asignatura

            df["COINCIDENCIAS"] = (
                (df["SIM_MASTER"] > 0.3).astype(int) +
                (df["SIM_UNIVERSIDAD"] > 0.3).astype(int) +
                (df["SIM_ASIGNATURA"] > 0.3).astype(int)
            )

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

                asignaturas_reconocidas = resultados["ASIGNATURA RECONOCIDA EN MUPEE"].nunique()
                porcentaje = (asignaturas_reconocidas / len(resultados)) * 100
                st.markdown(f"**📊 Porcentaje de asignaturas reconocidas respecto a coincidencias:** `{porcentaje:.1f}%`")
            else:
                st.warning("❗ No se encontraron coincidencias suficientes con los datos aportados.")
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
