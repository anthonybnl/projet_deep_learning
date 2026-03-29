import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from load_data import load_test_dataset, get_label_names
from network import get_model

LABEL_NAMES = get_label_names()

dataset = load_test_dataset()
N = len(dataset)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(device=DEVICE)

mean, std = 0.4979, 0.2450  # stats du dataset ChestMNIST (calculées sur les images d'entraînement)

# Préprocessing identique à l'entraînement du modèle hybride (ResNet → ImageNet stats)
inference_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),  # L → RGB
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[mean, mean, mean], std=[std, std, std]
        ),
    ]
)


def show_image(index: int):
    img, label = dataset[index]

    # img est une PIL Image (mode L, 64x64) → numpy array
    img_array = np.array(img)

    # Vecteur label : tableau numpy ou liste de 14 entiers
    label_vec = np.array(label).flatten()

    # Construction du texte des anomalies
    pathologies_presentes = [LABEL_NAMES[i] for i, v in enumerate(label_vec) if v == 1]

    if pathologies_presentes:
        label_text = "**Pathologies détectées :**\n" + "\n".join(
            f"- {p}" for p in pathologies_presentes
        )
    else:
        label_text = "**Aucune pathologie détectée** (image saine)"

    vecteur_text = "**Vecteur label brut :** `" + str(label_vec.tolist()) + "`"

    return img_array, label_text + "\n\n" + vecteur_text


def run_inference(index: int):
    img, _ = dataset[index]

    tensor = inference_transform(img).unsqueeze(0).to(DEVICE)  # (1, 3, 64, 64)

    with torch.no_grad():
        logits = model(tensor)  # (1, 14)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    THRESHOLD = 0.5
    colors = ["#e05c5c" if p >= THRESHOLD else "#5c8be0" for p in probs]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(LABEL_NAMES[::-1], probs[::-1], color=colors[::-1], height=0.6)
    ax.axvline(THRESHOLD, color="gray", linestyle="--", linewidth=1, label=f"Seuil = {THRESHOLD}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilité")
    ax.set_title("Résultats de l'inférence")
    ax.legend(loc="lower right", fontsize=8)

    for bar, p in zip(bars, probs[::-1]):
        ax.text(min(p + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                f"{p:.1%}", va="center", fontsize=8,
                color="#c0392b" if p >= THRESHOLD else "#2c5f9e")

    plt.tight_layout()
    return fig


with gr.Blocks(title="Projet Deep Learning — ChestMNIST") as demo:
    gr.Markdown("# ChestMNIST — démonstrateur")
    gr.Markdown(
        "Sélectionnez une image du jeu de test pour afficher la radio "
        "et les pathologies associées."
    )

    with gr.Row():
        with gr.Column(scale=1):
            index_slider = gr.Slider(
                minimum=0,
                maximum=N - 1,
                step=1,
                value=21930,
                label=f"Index de l'image (0 – {N - 1})",
            )
            btn = gr.Button("Afficher")

        with gr.Column(scale=2):
            image_out = gr.Image(label="Radio thoracique", type="numpy")
            label_out = gr.Markdown()

    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1):
            infer_btn = gr.Button("Inférence", variant="primary")
        with gr.Column(scale=3):
            pass

    infer_out = gr.Plot(label="Résultats de l'inférence")

    btn.click(fn=show_image, inputs=index_slider, outputs=[image_out, label_out])
    infer_btn.click(fn=run_inference, inputs=index_slider, outputs=infer_out)

    # Affichage initial
    demo.load(fn=show_image, inputs=index_slider, outputs=[image_out, label_out])


if __name__ == "__main__":
    demo.launch()
