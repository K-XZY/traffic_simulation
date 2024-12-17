import torch 
import numpy as np
import matplotlib.pyplot as plt

def plot_equation():
    V_max = 10
    L = 2
    epsilon = 0.0001
    D_critical = 10
    b = 0.0 

    D = np.linspace(L,D_critical,1000)


    C = V_max / (np.log(D_critical/L)) # the constant coefficient 
    future_V = C * np.log((D-b)/(L))
    print(future_V.shape)
    print(D.shape)

    a = np.linspace(0,L,1000)
    A = 0*a

    b = np.linspace(D_critical,D_critical + L, 1000)
    B = V_max*b/b


    fig = plt.figure(figsize = (16,9))
    plt.plot(a,A, label = 'completely stopped (collision)')
    plt.plot(D,future_V, label = 'f (reaction)')
    plt.plot(b,B, label = 'max velocity (free driving)')

    plt.title("The velocity reaction curve with respect to gap size D")
    plt.xlabel("D (m)")
    plt.ylabel("V (ms^-1)")
    plt.grid(True)
    plt.legend()

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
def plot_vector_field():
# Given parameters
    V_max = 10
    L = 2
    epsilon = 0.0001
    D_critical = 10
    b = 0.0

# Treat C as a free parameter
    D_min, D_max = L, D_critical + 10
    C_min, C_max = 0.1, 20.0

    num_points = 30
    D_values = np.linspace(D_min, D_max, num_points)
    C_values = np.linspace(C_min, C_max, num_points)

    D_mesh, C_mesh = np.meshgrid(D_values, C_values)

# Define V(D,C) = C * log((D - b)/L)
    V_mesh = C_mesh * np.log((D_mesh - b) / L)

# Compute numerical gradients
    dV_dC, dV_dD = np.gradient(V_mesh, C_values, D_values)

# Compute gradient magnitude for coloring
    magnitude = np.sqrt(dV_dD**2 + dV_dC**2)

    fig, ax = plt.subplots(figsize=(16, 9))

# Plot only the quiver field, no contour
# scale determines how large the arrows are. Increase scale to make them smaller.
    Q = ax.quiver(D_mesh, C_mesh, dV_dD, dV_dC, magnitude, 
                  cmap='plasma', scale=100, pivot='mid')

    ax.set_title("Gradient Field of V(D,C)")
    ax.set_xlabel("D (m)")
    ax.set_ylabel("C")
    ax.grid(True)

# Add a colorbar to show the magnitude scale
    cbar = fig.colorbar(Q, ax=ax, label='Gradient Magnitude')

    plt.show()

def plot_vector_field2():
# Parameters
    V_max = 10
    L = 2
    b = 0.0

# Define ranges for D and D_c
    D_min, D_max = L, 50  # D must be > L
    Dc_min, Dc_max = L + 5, 25  # D_c must be > L

    num_points = 30
    D_values = np.linspace(D_min, D_max, num_points)
    Dc_values = np.linspace(Dc_min, Dc_max, num_points)

    D_mesh, Dc_mesh = np.meshgrid(D_values, Dc_values)

# Define V(D, Dc)
    V_mesh = (V_max / np.log(Dc_mesh / L)) * np.log((D_mesh - b) / L)

# Compute gradients numerically
    dV_dDc, dV_dD = np.gradient(V_mesh, Dc_values, D_values)

# Compute gradient magnitude for coloring
    magnitude = np.sqrt(dV_dD**2 + dV_dDc**2)

# Plot quiver field
    fig, ax = plt.subplots(figsize=(16, 9))

    Q = ax.quiver(D_mesh, Dc_mesh, dV_dD, dV_dDc, magnitude, 
                  cmap='plasma', scale=70, pivot='mid')  # Adjust scale for smaller arrows

# Add colorbar
    cbar = fig.colorbar(Q, ax=ax, label='Gradient Magnitude')

# Labels and title
    ax.set_title("Gradient Field of V(D, Dc)")
    ax.set_xlabel("D (m)")
    ax.set_ylabel("D_c (m)")
    ax.grid(True)

    plt.show()

plot_vector_field2()
