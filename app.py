import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import DataLoader
from model_utils import ModelFeatureManager, get_all_cultivos_from_db

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicción Agrícola",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f8f0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


class AgriculturalApp:
    def __init__(self):
        self.model_manager = ModelFeatureManager()
        self.data_loader = DataLoader()
        self.model_data = None
        self.load_model_and_data()

    def load_model_and_data(self):
        """Carga el modelo y los datos"""
        try:
            # Cargar modelo
            self.model_data = self.model_manager.load_model_with_features(
                'models/agricultural_production_model.pkl'
            )

            # Cargar datos para visualizaciones
            self.clima_df, self.siembra_df, self.cosecha_df, self.cultivo_df = self.data_loader.load_all_data()
            self.prepare_visualization_data()

            # Obtener lista completa de cultivos
            self.all_cultivos = get_all_cultivos_from_db(self.data_loader)

            st.success("✅ Modelo y datos cargados correctamente")

        except Exception as e:
            st.error(f"❌ Error cargando el modelo o datos: {e}")
            st.info("Por favor, ejecuta primero el script de entrenamiento.")

    def prepare_visualization_data(self):
        """Prepara datos para visualizaciones"""
        if self.cosecha_df is not None and self.siembra_df is not None:
            self.datos_combinados = self.siembra_df.merge(
                self.cosecha_df,
                on=['id_region', 'id_cultivo', 'anio'],
                suffixes=('_siembra', '_cosecha')
            ).merge(
                self.cultivo_df,
                on='id_cultivo'
            )

            # Calcular rendimiento
            self.datos_combinados['rendimiento'] = (
                    self.datos_combinados['toneladas'] / self.datos_combinados['hectareas']
            )

    def render_sidebar(self):
        """Renderiza la barra lateral"""
        st.sidebar.title("🌱 Navegación")
        app_mode = st.sidebar.selectbox(
            "Selecciona una sección:",
            ["🏠 Dashboard", "📊 Análisis de Datos", "🔮 Predicciones", "📈 Tendencias", "ℹ️ Info del Modelo"]
        )

        st.sidebar.markdown("---")

        # Información del modelo cargado
        if self.model_data is not None:
            st.sidebar.info(f"""
            **Modelo Cargado:**
            - ✅ Características: {len(self.model_manager.expected_features)}
            - ✅ Cultivos: {len(self.model_manager.cultivo_columns)}
            - ✅ Regiones: {len(self.model_manager.region_columns)}
            """)

        st.sidebar.markdown("---")
        st.sidebar.info("Sistema de predicción de producción agrícola")

        return app_mode

    def render_model_info(self):
        """Muestra información detallada del modelo"""
        st.title("ℹ️ Información del Modelo")

        if self.model_data is None:
            st.warning("No se pudo cargar la información del modelo.")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total de Características", len(self.model_manager.expected_features))

        with col2:
            st.metric("Cultivos en el Modelo", len(self.model_manager.cultivo_columns))

        with col3:
            st.metric("Regiones en el Modelo", len(self.model_manager.region_columns))

        st.markdown("---")

        # Mostrar cultivos disponibles en el modelo
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌾 Cultivos en el Modelo")
            cultivos_modelo = [col.replace('cultivo_', '') for col in self.model_manager.cultivo_columns]
            st.write(f"**{len(cultivos_modelo)} cultivos:**")
            for cultivo in sorted(cultivos_modelo)[:20]:
                st.write(f"- {cultivo}")
            if len(cultivos_modelo) > 20:
                st.write(f"... y {len(cultivos_modelo) - 20} más")

        with col2:
            st.subheader("🏞️ Regiones en el Modelo")
            regiones_modelo = [col.replace('region_', '') for col in self.model_manager.region_columns]
            st.write(f"**{len(regiones_modelo)} regiones:**")
            st.write(", ".join(sorted(regiones_modelo)))

        # Características numéricas
        st.subheader("📊 Características Numéricas")
        st.write(f"**{len(self.model_manager.numeric_columns)} características:**")
        st.write(", ".join(self.model_manager.numeric_columns))

    def render_predictions(self):
        """Sección de predicciones corregida"""
        st.title("🔮 Predicción de Producción Agrícola")

        if self.model_data is None:
            st.error("❌ El modelo no está cargado. No se pueden hacer predicciones.")
            return

        st.success("✅ Modelo cargado correctamente. Puedes realizar predicciones.")

        # Mostrar advertencia sobre cultivos no vistos
        if hasattr(self, 'all_cultivos'):
            cultivos_modelo = [col.replace('cultivo_', '') for col in self.model_manager.cultivo_columns]
            cultivos_no_en_modelo = [c for c in self.all_cultivos if c not in cultivos_modelo]

            if cultivos_no_en_modelo:
                st.warning(f"""
                **Nota:** {len(cultivos_no_en_modelo)} cultivos de la base de datos no están en el modelo entrenado.
                Solo puedes predecir con los {len(cultivos_modelo)} cultivos del modelo.
                """)

        # Formulario de entrada
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📋 Parámetros de Siembra")

                # Selector de región - SOLO LAS DEL MODELO
                region_options = [int(col.replace('region_', ''))
                                  for col in self.model_manager.region_columns]
                region = st.selectbox(
                    "Región *",
                    options=sorted(region_options),
                    help="Regiones disponibles en el modelo entrenado"
                )

                # Selector de cultivo - SOLO LOS DEL MODELO
                cultivo_options = [col.replace('cultivo_', '')
                                   for col in self.model_manager.cultivo_columns]
                cultivo = st.selectbox(
                    "Cultivo *",
                    options=sorted(cultivo_options),
                    help="Cultivos disponibles en el modelo entrenado"
                )

                hectareas = st.number_input(
                    "Hectáreas sembradas *",
                    min_value=1.0,
                    max_value=10000.0,
                    value=100.0,
                    step=10.0
                )

            with col2:
                st.subheader("🌤️ Condiciones Climáticas Esperadas")

                precip_total = st.slider(
                    "Precipitación total anual (mm) *",
                    min_value=0.0,
                    max_value=2000.0,
                    value=800.0,
                    step=50.0
                )

                temp_max_promedio = st.slider(
                    "Temperatura máxima promedio (°C) *",
                    min_value=10.0,
                    max_value=40.0,
                    value=25.0,
                    step=1.0
                )

                temp_min_promedio = st.slider(
                    "Temperatura mínima promedio (°C) *",
                    min_value=0.0,
                    max_value=25.0,
                    value=12.0,
                    step=1.0
                )

                dias_helada = st.slider(
                    "Días con helada esperados *",
                    min_value=0,
                    max_value=100,
                    value=5
                )

            st.markdown("**\* Campos requeridos**")

            # Botón de predicción
            submitted = st.form_submit_button("🎯 Predecir Producción", use_container_width=True)

            if submitted:
                self.make_prediction(
                    region, cultivo, hectareas,
                    precip_total, temp_max_promedio, temp_min_promedio, dias_helada
                )

    def make_prediction(self, region, cultivo, hectareas,
                        precip_total, temp_max_promedio, temp_min_promedio, dias_helada):
        """Realiza la predicción usando el método corregido"""
        try:
            # Preparar datos de entrada
            input_data = pd.DataFrame({
                'hectareas': [hectareas],
                'precip_total': [precip_total],
                'temp_max_promedio': [temp_max_promedio],
                'temp_min_promedio': [temp_min_promedio],
                'dias_helada': [dias_helada]
            })

            # Preparar características para predicción
            prediction_features = self.model_manager.prepare_prediction_features(
                input_data, cultivo, region
            )

            # Realizar predicción
            X_pred_scaled = self.model_data['scaler'].transform(prediction_features)
            prediction = self.model_data['model'].predict(X_pred_scaled)[0]

            # Calcular rendimiento
            rendimiento = prediction / hectareas

            # Mostrar resultados
            self.display_prediction_results(prediction, rendimiento, hectareas, cultivo, region)

        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")

    def display_prediction_results(self, prediction, rendimiento, hectareas, cultivo, region):
        """Muestra los resultados de la predicción"""
        st.markdown("---")
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

        st.subheader("📊 Resultados de la Predicción")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Producción Predicha",
                f"{prediction:,.2f} ton",
                delta=f"Rendimiento: {rendimiento:.2f} ton/ha"
            )

        with col2:
            st.metric("Área Sembrada", f"{hectareas:,.1f} ha")

        with col3:
            st.metric("Eficiencia", f"{(rendimiento / hectareas * 1000):.1f}%" if hectareas > 0 else "N/A")

        # Información adicional
        st.write(f"**Cultivo:** {cultivo}")
        st.write(f"**Región:** {region}")
        st.write(f"**Rendimiento estimado:** {rendimiento:.2f} toneladas por hectárea")

        st.markdown('</div>', unsafe_allow_html=True)

        # Gráficos de resultados
        self.render_prediction_charts(prediction, rendimiento, hectareas)

    def render_prediction_charts(self, prediction, rendimiento, hectareas):
        """Renderiza gráficos para los resultados de predicción"""
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de gauge para rendimiento
            if hasattr(self, 'datos_combinados') and not self.datos_combinados.empty:
                rendimiento_promedio = self.datos_combinados['rendimiento'].mean()

                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=rendimiento,
                    delta={'reference': rendimiento_promedio},
                    gauge={
                        'axis': {'range': [None, max(rendimiento * 1.5, rendimiento_promedio * 1.5)]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, rendimiento_promedio], 'color': "lightgray"},
                            {'range': [rendimiento_promedio, rendimiento_promedio * 1.5], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': rendimiento_promedio
                        }
                    },
                    title={'text': "Rendimiento vs Promedio Histórico"}
                ))

                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Gráfico de barras simple
            fig = px.bar(
                x=['Producción Predicha'],
                y=[prediction],
                title="Producción Total Estimada",
                labels={'x': '', 'y': 'Toneladas'},
                color_discrete_sequence=['#2E8B57']
            )
            st.plotly_chart(fig, use_container_width=True)
    def render_dashboard(self):
        """Renderiza el dashboard principal"""
        st.markdown('<h1 class="main-header">🌾 Dashboard de Producción Agrícola</h1>', unsafe_allow_html=True)

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_cosecha = self.datos_combinados['toneladas'].sum()
            st.metric("Producción Total", f"{total_cosecha:,.0f} ton")

        with col2:
            total_hectareas = self.datos_combinados['hectareas'].sum()
            st.metric("Área Sembrada Total", f"{total_hectareas:,.0f} ha")

        with col3:
            rendimiento_promedio = self.datos_combinados['rendimiento'].mean()
            st.metric("Rendimiento Promedio", f"{rendimiento_promedio:.2f} ton/ha")

        with col4:
            cultivos_unicos = self.datos_combinados['nombre_cultivo'].nunique()
            st.metric("Tipos de Cultivo", cultivos_unicos)

        st.markdown("---")

        # Gráficos principales
        col1, col2 = st.columns(2)

        with col1:
            self.render_production_trend()

        with col2:
            self.render_crop_distribution()

        col3, col4 = st.columns(2)

        with col3:
            self.render_yield_by_region()

        with col4:
            self.render_climate_impact()


    def render_data_analysis(self):
        """Sección de análisis de datos"""
        st.title("📊 Análisis Exploratorio de Datos")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Datos de Siembra")
            st.dataframe(
                self.siembra_df.head(100),
                use_container_width=True,
                height=300
            )

            # Resumen estadístico
            st.subheader("Estadísticas de Siembra")
            st.dataframe(self.siembra_df[['hectareas', 'anio']].describe())

        with col2:
            st.subheader("Datos de Cosecha")
            st.dataframe(
                self.cosecha_df.head(100),
                use_container_width=True,
                height=300
            )

            # Resumen estadístico
            st.subheader("Estadísticas de Cosecha")
            st.dataframe(self.cosecha_df[['toneladas', 'anio']].describe())

        st.markdown("---")

        # Análisis de correlación
        st.subheader("🔍 Análisis de Correlaciones")

        # Preparar datos para correlación
        datos_analisis = self.datos_combinados[['hectareas', 'toneladas', 'rendimiento', 'anio']].copy()

        # Calcular matriz de correlación
        corr_matrix = datos_analisis.corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Matriz de Correlación"
        )

        st.plotly_chart(fig, use_container_width=True)
    def render_production_trend(self):
        """Gráfico de tendencia de producción"""
        st.subheader("📈 Tendencia de Producción Anual")

        produccion_anual = self.datos_combinados.groupby('anio').agg({
            'toneladas': 'sum',
            'hectareas': 'sum'
        }).reset_index()

        produccion_anual['rendimiento'] = produccion_anual['toneladas'] / produccion_anual['hectareas']

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=produccion_anual['anio'], y=produccion_anual['toneladas'],
                       name="Producción (ton)", line=dict(color='#2E8B57', width=3)),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=produccion_anual['anio'], y=produccion_anual['rendimiento'],
                       name="Rendimiento (ton/ha)", line=dict(color='#FF6B6B', width=3)),
            secondary_y=True,
        )

        fig.update_layout(
            title="Evolución de la Producción y Rendimiento",
            xaxis_title="Año",
            hovermode='x unified',
            height=400
        )

        fig.update_yaxes(title_text="Producción (ton)", secondary_y=False)
        fig.update_yaxes(title_text="Rendimiento (ton/ha)", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)
    def render_crop_distribution(self):
        """Gráfico de distribución de cultivos"""
        st.subheader("🥦 Distribución de Cultivos")

        cultivo_produccion = self.datos_combinados.groupby('nombre_cultivo').agg({
            'toneladas': 'sum',
            'hectareas': 'sum'
        }).reset_index()

        cultivo_produccion['rendimiento'] = cultivo_produccion['toneladas'] / cultivo_produccion['hectareas']

        fig = px.sunburst(
            cultivo_produccion,
            path=['nombre_cultivo'],
            values='toneladas',
            title="Distribución de Producción por Cultivo",
            color='rendimiento',
            color_continuous_scale='Viridis',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
    def render_yield_by_region(self):
        """Gráfico de rendimiento por región"""
        st.subheader("🏞️ Rendimiento por Región")

        region_rendimiento = self.datos_combinados.groupby('id_region').agg({
            'rendimiento': 'mean',
            'toneladas': 'sum'
        }).reset_index()

        fig = px.bar(
            region_rendimiento,
            x='id_region',
            y='rendimiento',
            title="Rendimiento Promedio por Región",
            color='rendimiento',
            color_continuous_scale='Blues',
            height=400
        )

        fig.update_layout(xaxis_title="Región", yaxis_title="Rendimiento (ton/ha)")

        st.plotly_chart(fig, use_container_width=True)

    def render_climate_impact(self):
        """Gráfico de impacto climático"""
        st.subheader("🌤️ Análisis Climático")

        # Agrupar datos climáticos por año
        clima_anual = self.clima_df.groupby('anio').agg({
            'precipitacion': 'mean',
            'temperatura_max': 'mean',
            'temperatura_min': 'mean'
        }).reset_index()

        # Combinar con datos de producción
        clima_produccion = clima_anual.merge(
            self.datos_combinados.groupby('anio')['rendimiento'].mean().reset_index(),
            on='anio'
        )

        fig = px.scatter(
            clima_produccion,
            x='precipitacion',
            y='rendimiento',
            size='temperatura_max',
            color='temperatura_max',
            title="Relación Precipitación vs Rendimiento",
            trendline="lowess",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_data_analysis(self):
        """Sección de análisis de datos"""
        st.title("📊 Análisis Exploratorio de Datos")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Datos de Siembra")
            st.dataframe(
                self.siembra_df.head(100),
                use_container_width=True,
                height=300
            )

            # Resumen estadístico
            st.subheader("Estadísticas de Siembra")
            st.dataframe(self.siembra_df[['hectareas', 'anio']].describe())

        with col2:
            st.subheader("Datos de Cosecha")
            st.dataframe(
                self.cosecha_df.head(100),
                use_container_width=True,
                height=300
            )

            # Resumen estadístico
            st.subheader("Estadísticas de Cosecha")
            st.dataframe(self.cosecha_df[['toneladas', 'anio']].describe())

        st.markdown("---")

        # Análisis de correlación
        st.subheader("🔍 Análisis de Correlaciones")

        # Preparar datos para correlación
        datos_analisis = self.datos_combinados[['hectareas', 'toneladas', 'rendimiento', 'anio']].copy()

        # Calcular matriz de correlación
        corr_matrix = datos_analisis.corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Matriz de Correlación"
        )

        st.plotly_chart(fig, use_container_width=True)
    def render_prediction_charts(self, prediction, rendimiento, hectareas):
        """Renderiza gráficos para los resultados de predicción"""
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de comparación con promedio histórico
            rendimiento_promedio = self.datos_combinados['rendimiento'].mean()

            fig = go.Figure()

            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=rendimiento,
                delta={'reference': rendimiento_promedio},
                gauge={
                    'axis': {'range': [None, max(rendimiento * 1.5, rendimiento_promedio * 1.5)]},
                    'steps': [
                        {'range': [0, rendimiento_promedio], 'color': "lightgray"},
                        {'range': [rendimiento_promedio, rendimiento_promedio * 1.5], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': rendimiento_promedio
                    }
                },
                title={'text': "Rendimiento vs Promedio Histórico"}
            ))

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Gráfico de desglose
            labels = ['Producción Esperada', 'Área No Productiva']
            values = [prediction, hectareas * rendimiento - prediction]

            fig = px.pie(
                values=values,
                names=labels,
                title="Distribución de Producción Esperada",
                color=labels,
                color_discrete_map={
                    'Producción Esperada': '#2E8B57',
                    'Área No Productiva': '#FF6B6B'
                }
            )

            st.plotly_chart(fig, use_container_width=True)

    def render_trends(self):
        """Sección de análisis de tendencias"""
        st.title("📈 Análisis de Tendencias y Proyecciones")

        st.warning("""
        Esta sección utiliza datos históricos para identificar tendencias 
        y realizar proyecciones futuras. Las predicciones son estimaciones 
        basadas en patrones históricos.
        """)

        # Selectores para análisis
        col1, col2, col3 = st.columns(3)

        with col1:
            cultivo_tendencia = st.selectbox(
                "Selecciona cultivo para análisis:",
                options=sorted(self.cultivo_df['nombre_cultivo'].unique()),
                key="cultivo_trend"
            )

        with col2:
            region_tendencia = st.selectbox(
                "Selecciona región:",
                options=sorted(self.siembra_df['id_region'].unique()),
                key="region_trend"
            )

        with col3:
            metricas = st.selectbox(
                "Métrica a analizar:",
                ["Producción", "Rendimiento", "Área Sembrada"]
            )

        # Gráfico de tendencia
        self.render_trend_analysis(cultivo_tendencia, region_tendencia, metricas)

        # Proyección futura
        st.subheader("🔭 Proyección para Próximos Años")

        anos_proyeccion = st.slider(
            "Años a proyectar:",
            min_value=1,
            max_value=10,
            value=5
        )

        if st.button("Generar Proyección", use_container_width=True):
            self.render_projection(cultivo_tendencia, region_tendencia, anos_proyeccion)

    def render_trend_analysis(self, cultivo, region, metrica):
        """Renderiza análisis de tendencias"""
        # Filtrar datos
        cultivo_id = self.cultivo_df[
            self.cultivo_df['nombre_cultivo'] == cultivo
            ]['id_cultivo'].iloc[0]

        datos_filtrados = self.datos_combinados[
            (self.datos_combinados['id_region'] == region) &
            (self.datos_combinados['id_cultivo'] == cultivo_id)
            ]

        if datos_filtrados.empty:
            st.warning("No hay datos suficientes para el análisis de tendencias con los filtros seleccionados.")
            return

        # Seleccionar métrica
        if metrica == "Producción":
            columna = 'toneladas'
            titulo = f"Tendencia de Producción - {cultivo}"
        elif metrica == "Rendimiento":
            columna = 'rendimiento'
            titulo = f"Tendencia de Rendimiento - {cultivo}"
        else:
            columna = 'hectareas'
            titulo = f"Tendencia de Área Sembrada - {cultivo}"

        fig = px.scatter(
            datos_filtrados,
            x='anio',
            y=columna,
            trendline="lowess",
            title=titulo,
            size='toneladas',
            color='rendimiento',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_projection(self, cultivo, region, anos_proyeccion):
        """Renderiza proyección futura"""
        # Aquí iría la lógica para proyecciones más sofisticadas
        # Por ahora, mostramos un gráfico placeholder

        st.info("""
        ⚠️ Las proyecciones mostradas son estimaciones basadas en tendencias históricas 
        y no consideran cambios climáticos extremos o eventos imprevistos.
        """)

        # Placeholder para proyección
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[2023, 2024, 2025, 2026, 2027],
            y=[100, 120, 115, 130, 125],
            mode='lines+markers',
            name='Proyección',
            line=dict(dash='dash', color='orange')
        ))

        fig.update_layout(
            title=f"Proyección de Producción - {cultivo}",
            xaxis_title="Año",
            yaxis_title="Producción (ton)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
def main():
    app = AgriculturalApp()
    app_mode = app.render_sidebar()

    if app_mode == "🏠 Dashboard":
        app.render_dashboard()
    elif app_mode == "📊 Análisis de Datos":
        app.render_data_analysis()
    elif app_mode == "🔮 Predicciones":
        app.render_predictions()
    elif app_mode == "📈 Tendencias":
        app.render_trends()
    elif app_mode == "ℹ️ Info del Modelo":
        app.render_model_info()

if __name__ == "__main__":
    main()