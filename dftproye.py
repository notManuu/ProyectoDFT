import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="DFT: Análisis", layout="wide")

# Estilos CSS
st.markdown("""
<style>
    .header-style { font-size: 22px; font-weight: bold; color: #1f77b4; margin-top: 20px;}
    .equation-box { background-color: #f0f8ff; padding: 15px; border-radius: 5px; border-left: 5px solid #1f77b4; margin-bottom: 20px;}
    .result-box { background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 5px; text-align: center; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📊 Análisis DFT")
    st.markdown("""
    Cálculo de DFT con desglose analítico, para obtener las amplitudes.
    """)

    # --- 1. CONFIGURACIÓN (SIDEBAR) ---
    st.sidebar.header("1. Parámetros")
    
    # Frecuencia de Muestreo
    Fs = st.sidebar.number_input("Frecuencia de Muestreo (fs) [Hz]", value=1.0, min_value=0.1, step=0.1, format="%.2f")
    Ts = 1 / Fs 
    
    # --- SELECCIÓN DE DATOS ---
    st.sidebar.divider()
    st.sidebar.subheader("Selección de Datos")
    
    data_source = st.sidebar.radio(
        "Fuente de datos:",
        ("Ingreso Manual", "Ejercicio 1 (Tabla 1)", "Ejercicio 2 (Tabla 2)")
    )

    # Definición de datos pre-cargados
    # Tabla 1
    data_ej1 = [
        127, 254.3192037, 196.6237684, 129.3229379, 102.932, 32.12134611,
        17.01864601, 19.6538024, -127, -254.3192037, -196.6237684,
        -129.3229379, -102.932, -32.12134611, -17.01864601, -19.6538024
    ]

    # Tabla 2
    data_ej2 = [
        109.3223305, 189.3587753, 193.9985207, 169.7603309, 124.3223305,
        39.89466492, -10.60660172, -31.77672342, -109.3223305, -189.3587753,
        -193.9985207, -169.7603309, -124.3223305, -39.89466492, 10.60660172,
        31.77672342
    ]

    x_n = []

    # Lógica de selección
    if data_source == "Ingreso Manual":
        default_data = "0, 169.33, 119.73, 169.33, 0, -169.33, -119.73, -169.33"
        raw_input = st.sidebar.text_area("Valores de x(n) [separados por coma]", default_data, height=120)
        try:
            x_n = np.array([float(val.strip()) for val in raw_input.split(',') if val.strip()])
            # --- AQUÍ ESTÁ EL CAMBIO: Mensaje de confirmación para manual ---
            st.sidebar.success(f"✅ Datos manuales cargados. (N={len(x_n)})")
        except:
            st.error("Error en el formato de los datos manuales. Asegúrate de usar comas para separar.")
            st.stop()
            
    elif data_source == "Ejercicio 1 (Tabla 1)":
        x_n = np.array(data_ej1)
        st.sidebar.info(f"✅ Datos Ejercicio 1 cargados. (N={len(x_n)})")
    else:
        x_n = np.array(data_ej2)
        st.sidebar.info(f"✅ Datos Ejercicio 2 cargados. (N={len(x_n)})")

    # Procesamiento común
    N = len(x_n)
    n = np.arange(N)
    t = n * Ts 

    # Selección de Armónico
    st.sidebar.divider()
    m_select = st.sidebar.number_input("Seleccionar Armónico (m) para evaluar", 0, N-1, 1)

    # --- SECCIÓN I: DATOS DE ENTRADA (SIN TIEMPO) ---
    st.markdown('<div class="header-style">I. Señal Discreta de Entrada</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.caption("Tabla de Muestras")
        df_input = pd.DataFrame({
            "n (Índice)": n, 
            "Amplitud x(n)": x_n
        })
        st.dataframe(df_input, use_container_width=True, hide_index=True)
    
    with col2:
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=n, y=x_n, mode='lines+markers', name='x(n)', line=dict(color='#2E86C1')))
        fig_t.update_layout(title="Secuencia de Entrada x(n)", xaxis_title="Muestra n", yaxis_title="Amplitud", height=250)
        st.plotly_chart(fig_t, use_container_width=True)

    # --- SECCIÓN II: DESGLOSE ANALÍTICO CON NORMALIZACIÓN ---
    st.markdown(f'<div class="header-style">II. Desglose Analítico (m={m_select})</div>', unsafe_allow_html=True)
    st.markdown("Cálculo de correlación y aplicación del factor de escala $1/(N/2)$.")
    
    # 1. Cálculos base (Correlación)
    theta = (2 * np.pi * m_select * n) / N
    cos_vals = np.cos(theta)
    sin_vals = np.sin(theta)
    prod_cos = x_n * cos_vals
    prod_sin = x_n * sin_vals
    
    # 2. Tabla detallada
    df_detail = pd.DataFrame({
        "n": n,
        "x(n)": x_n,
        f"cos(...)": cos_vals,
        "x(n)·cos": prod_cos,
        f"sen(...)": sin_vals,
        "x(n)·sen": prod_sin
    })
    st.dataframe(df_detail.style.format("{:.4f}"), use_container_width=True)

    # 3. Sumatorias y Normalización
    sum_real = np.sum(prod_cos)
    sum_imag = np.sum(prod_sin)
    
    norm_factor = 1 / (N / 2)
    val_real_norm = sum_real * norm_factor
    val_imag_norm = sum_imag * norm_factor
    
    # Magnitud y Fase
    mag_final = np.sqrt(val_real_norm**2 + val_imag_norm**2)
    phase_rad = np.arctan2(-val_imag_norm, val_real_norm)
    phase_deg = np.degrees(phase_rad)

    # --- VISUALIZACIÓN MATEMÁTICA DE RESULTADOS ---
    st.subheader("Resultados de la Sumatoria")
    
    col_math1, col_math2 = st.columns(2)

    with col_math1:
        st.markdown("**Parte Real (Cosenos):**")
        # Formula Sumatoria Cruda
        st.latex(r"\sum_{n=0}^{N-1} x[n] \cdot \cos\left(\frac{2\pi m n}{N}\right) = " + f"{sum_real:.4f}")
        # Formula Normalizada
        st.latex(r"\text{Real}_{norm} = \frac{1}{N/2} \sum (\dots) = " + f"{val_real_norm:.4f}")

    with col_math2:
        st.markdown("**Parte Imaginaria (Senos):**")
        # Formula Sumatoria Cruda
        st.latex(r"\sum_{n=0}^{N-1} x[n] \cdot \sin\left(\frac{2\pi m n}{N}\right) = " + f"{sum_imag:.4f}")
        # Formula Normalizada
        st.latex(r"\text{Imag}_{norm} = \frac{1}{N/2} \sum (\dots) = " + f"{val_imag_norm:.4f}")
    
    st.divider()
    
    # Resultado Final en un bloque destacado
    c_res1, c_res2 = st.columns(2)
    with c_res1:
        st.success(f"**Magnitud Normalizada (Amplitud Real)**")
        st.latex(r"|X|_{norm} = \sqrt{Re_{norm}^2 + Im_{norm}^2} = " + f" {mag_final:.4f}")
    with c_res2:
        st.info(f"**Fase**")
        st.latex(r"\phi = \arctan\left(\frac{-Im}{Re}\right) = " + f"{phase_deg:.2f}^\circ")

    # --- SECCIÓN III: EVALUACIÓN CON MAGNITUD NORMALIZADA ---
    st.markdown(f'<div class="header-style">III. Evaluación Temporal: v(t)</div>', unsafe_allow_html=True)
    
    # Frecuencia del armónico
    freq_hz = m_select * (Fs / N)
    
    # Mostrar la Ecuación usando la magnitud normalizada
    st.markdown('<div class="equation-box">', unsafe_allow_html=True)
    st.markdown("**Ecuación de reconstrucción del armónico:**")
    st.latex(r"v_{" + str(m_select) + r"}(t) = \text{Magnitud}_{norm} \cdot \cos(\omega t + \phi)")
    st.markdown("**Sustituyendo los valores obtenidos:**")
    st.latex(rf"v_{{{m_select}}}(t) = {mag_final:.4f} \cdot \cos(2\pi \cdot {freq_hz:.2f} \cdot t + {phase_deg:.2f}^\circ)")
    st.markdown('</div>', unsafe_allow_html=True)

    # Evaluación numérica
    v_evaluated = mag_final * np.cos(2 * np.pi * freq_hz * t + phase_rad)
    
    col_h1, col_h2 = st.columns([1, 1])
    
    with col_h1:
        st.subheader("Tabla de Valores Evaluados")
        df_eval = pd.DataFrame({
            "n": n,
            "t (seg)": t,
            f"v_{m_select}(t)": v_evaluated
        })
        st.dataframe(df_eval.style.format({"t (seg)": "{:.4f}", f"v_{m_select}(t)": "{:.4f}"}), use_container_width=True)
        
    with col_h2:
        st.subheader("Gráfica del Armónico")
        fig_h = go.Figure()
        # Dibujamos curva suave de referencia
        t_fine = np.linspace(0, t[-1], 200)
        v_fine = mag_final * np.cos(2 * np.pi * freq_hz * t_fine + phase_rad)
        fig_h.add_trace(go.Scatter(x=t_fine, y=v_fine, mode='lines', name='Continua', line=dict(color='lightgray', dash='dash')))
        # Puntos evaluados
        fig_h.add_trace(go.Scatter(x=t, y=v_evaluated, mode='markers', name='Evaluados', marker=dict(size=10, color='orange')))
        
        fig_h.update_layout(xaxis_title="Tiempo (s)", yaxis_title="Amplitud (v)", height=300)
        st.plotly_chart(fig_h, use_container_width=True)

    # --- SECCIÓN IV: COMPROBACIÓN FINAL ---
    st.markdown('<div class="header-style">IV. Comprobación (Síntesis)</div>', unsafe_allow_html=True)
    st.caption("Verificación matemática usando FFT estándar (Suma de todos los armónicos).")

    X_fft = np.fft.fft(x_n)
    x_reconst = np.fft.ifft(X_fft).real

    df_check = pd.DataFrame({
        "n": n,
        "Original": x_n,
        "Reconstruida": x_reconst,
        "Error": np.abs(x_n - x_reconst)
    })
    st.dataframe(df_check.style.format("{:.4f}"), use_container_width=True)

if __name__ == "__main__":
    main()