# LHC_simulator.py
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


def detector_evento(tipo="electron", size=64):
    img = np.zeros((size, size))
    x = np.linspace(0, size-1, size)

    if tipo == "electron":
        m = np.random.uniform(-1, 1)
        b = np.random.randint(0, size)
        y = m*x + b
        coords = np.vstack([x, y]).astype(int).T

    elif tipo == "muon":
        m = np.random.uniform(-1, 1)
        b = np.random.randint(0, size)
        y = m*x + b + np.random.normal(0, 2, size)
        coords = np.vstack([x, y]).astype(int).T

    elif tipo == "proton":
        vueltas = np.random.randint(2, 6)
        theta = np.linspace(0, vueltas*np.pi, size*3)
        r = np.linspace(5, size/2, size*3) + np.random.uniform(-3,3)
        x = (r*np.cos(theta) + size/2 + np.random.uniform(-5,5)).astype(int)
        y = (r*np.sin(theta) + size/2 + np.random.uniform(-5,5)).astype(int)
        coords = np.vstack([x, y]).T

    elif tipo == "neutrino":
        a = np.random.uniform(0.01, 0.05)
        h = np.random.randint(10, size-10)
        k = np.random.randint(10, size-10)
        y = a*(x-h)**2 + k
        coords = np.vstack([x, y]).astype(int).T

    elif tipo == "quark":
        steps = np.random.randint(50, 150)
        x_pos = np.cumsum(np.random.choice([-1, 0, 1], steps))
        y_pos = np.cumsum(np.random.choice([-1, 0, 1], steps))
        x_pos = np.clip(x_pos + size//2, 0, size-1)
        y_pos = np.clip(y_pos + size//2, 0, size-1)
        coords = np.vstack([x_pos, y_pos]).T

    elif tipo == "tau":
        m = np.random.uniform(-0.2, 0.2)
        b = np.random.randint(0, size)
        y = m*x + b + np.random.normal(0, 0.5, size)
        coords = np.vstack([x, y]).astype(int).T

    coords = coords[(coords[:,0] >= 0) & (coords[:,0] < size) &
                    (coords[:,1] >= 0) & (coords[:,1] < size)]

    img[coords[:,0], coords[:,1]] = 1
    ruido = np.random.binomial(1, 0.01, (size, size))
    img = np.clip(img + ruido, 0, 1)

    return img


def experiment_CMS(output_dir="CMS_data", num_img=1000, size=64, zip_name="LHC_eventos"):
    clases = ["electron", "muon", "proton", "neutrino", "quark", "tau"]

    for c in clases:
        os.makedirs(os.path.join(output_dir, c), exist_ok=True)

    for c in clases:
        for i in range(num_img):
            img = detector_evento(c, size=size)
            plt.imsave(os.path.join(output_dir, c, f"{c}_{i}.png"), img, cmap="gray")

    shutil.make_archive(zip_name, 'zip', output_dir)
    print(f"Dataset generado con {num_img} imágenes por clase.")
    print(f"Archivo comprimido: {zip_name}.zip")
