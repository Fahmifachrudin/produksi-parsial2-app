import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

st.title("Analisis Turunan Parsial - Optimalisasi Keuntungan Produksi Krupuk")

# Simbol x dan y
x, y = sp.symbols('x y')

# Fungsi keuntungan K(x, y)
fungsi_str = "250*x + 150*y - 2*x**2 - y**2 - 3*x*y"
f = sp.sympify(fungsi_str)

# Turunan parsial
fx = sp.diff(f, x)
fy = sp.diff(f, y)

# Tampilkan fungsi dan turunannya
st.latex(f"K(x, y) = {sp.latex(f)}")
st.latex(f"\\frac{{\\partial K}}{{\\partial x}} = {sp.latex(fx)}")
st.latex(f"\\frac{{\\partial K}}{{\\partial y}} = {sp.latex(fy)}")

# Input nilai aktual x dan y
x0 = st.number_input("Jumlah Kaleng Kerupuk Putih (x):", value=10)
y0 = st.number_input("Jumlah Kaleng Kerupuk Drokdok (y):", value=7)

# Evaluasi nilai fungsi dan gradien
f_val = f.subs({x: x0, y: y0})
fx_val = fx.subs({x: x0, y: y0})
fy_val = fy.subs({x: x0, y: y0})

# Tampilkan hasil evaluasi
st.write("Nilai Keuntungan (dalam satuan tidak dikalikan ribuan):", f_val)
st.write(f"Gradien di titik ({x0}, {y0}): (∂K/∂x = {fx_val}, ∂K/∂y = {fy_val})")

# Visualisasi 3D: Fungsi dan bidang singgung
st.subheader("Visualisasi Permukaan Keuntungan dan Bidang Singgungnya")

x_vals = np.linspace(x0 - 5, x0 + 5, 50)
y_vals = np.linspace(y0 - 5, y0 + 5, 50)
X, Y = np.meshgrid(x_vals, y_vals)

# Fungsi lambdify untuk evaluasi numerik
K_func = sp.lambdify((x, y), f, 'numpy')
Z = K_func(X, Y)

# Bidang singgung
Z_tangent = float(f_val) + float(fx_val) * (X - x0) + float(fy_val) * (Y - y0)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis', label='Keuntungan')
ax.plot_surface(X, Y, Z_tangent, alpha=0.5, color='red', label='Bidang Singgung')
ax.set_title("Permukaan Fungsi Keuntungan & Bidang Singgung")
ax.set_xlabel('x (Kerupuk Putih)')
ax.set_ylabel('y (Kerupuk Drokdok)')
ax.set_zlabel('Keuntungan')
st.pyplot(fig)

# Tambahkan visualisasi gradien 2D (opsional)
st.subheader("Visualisasi Vektor Gradien 2D")
fig2, ax2 = plt.subplots()
ax2.quiver(x0, y0, fx_val, fy_val, angles='xy', scale_units='xy', scale=1, color='blue')
ax2.set_xlim([x0 - 5, x0 + 5])
ax2.set_ylim([y0 - 5, y0 + 5])
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Vektor Gradien dari Fungsi Keuntungan')
st.pyplot(fig2)
