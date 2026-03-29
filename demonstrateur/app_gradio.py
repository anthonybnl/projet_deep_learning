import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from load_data import load_test_dataset, get_label_names
from model_hybrid import get_model_hybrid
from model_AE import get_model_ae

LABEL_NAMES = get_label_names()

dataset = load_test_dataset()
N = len(dataset)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_hybrid = get_model_hybrid(device=DEVICE)
model_ae = get_model_ae(device=DEVICE)

mean, std = (
    0.4979,
    0.2450,
)  # stats du dataset train ChestMNIST

# transform pour le modèle hybride (3 canaux RGB, tensor, normalisation)
inference_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),  # L → RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std]),
    ]
)

# transform pour l'autoencodeur
ae_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ]
)

THRESHOLD_AE = 0.87  # percentile 85 calculé sur les images saines de validation


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


def evaluate_hybrid(index: int):
    img, _ = dataset[index]

    tensor = inference_transform(img).unsqueeze(0).to(DEVICE)  # (1, 3, 64, 64)

    with torch.no_grad():
        logits = model_hybrid(tensor)  # (1, 14)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    THRESHOLD = 0.5
    colors = ["#e05c5c" if p >= THRESHOLD else "#5c8be0" for p in probs]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(LABEL_NAMES[::-1], probs[::-1], color=colors[::-1], height=0.6)
    ax.axvline(
        THRESHOLD,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Seuil = {THRESHOLD}",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilité")
    ax.set_title("Résultats de l'inférence")
    ax.legend(loc="lower right", fontsize=8)

    for bar, p in zip(bars, probs[::-1]):
        ax.text(
            min(p + 0.02, 0.97),
            bar.get_y() + bar.get_height() / 2,
            f"{p:.1%}",
            va="center",
            fontsize=8,
            color="#c0392b" if p >= THRESHOLD else "#2c5f9e",
        )

    plt.tight_layout()
    return fig


def evaluate_ae(index: int):
    img, _ = dataset[index]

    tensor = ae_transform(img).unsqueeze(0).to(DEVICE)  # (1, 1, 64, 64)

    with torch.no_grad():
        recon, _ = model_ae(tensor)  # recon : (1, 1, 64, 64), valeurs ∈ [0,1]

    # Score d'anomalie : MSE entre reconstruction et image normalisée
    # (identique à compute_anomaly_scores dans le notebook)
    mse_per_pixel = torch.nn.functional.mse_loss(recon, tensor, reduction="none")
    mse = mse_per_pixel.view(1, -1).mean(dim=1).item()

    # Images numpy pour affichage
    original_np = np.array(img)  # uint8, HxW
    recon_np = recon[0, 0].cpu().numpy()  # float32 ∈ [0,1]

    anomalie = mse > THRESHOLD_AE
    ae_prediction = "Anomalie détectée" if anomalie else "Image saine"
    color = "#c0392b" if anomalie else "#27ae60"

    verdict_html = f"""
    <div style="
        background: {color}18;
        border: 2px solid {color};
        border-radius: 12px;
        padding: 20px 28px;
        text-align: center;
        font-family: sans-serif;
    ">
        <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{ae_prediction}</div>
        <div style="margin-top: 10px; font-size: 0.95rem; color: #555;">
            MSE = <strong>{mse:.4f}</strong> &nbsp;|&nbsp; seuil = <strong>{THRESHOLD_AE}</strong>
        </div>
    </div>
    """

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original_np, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Reconstruction (MSE = {mse:.4f})")
    axes[1].axis("off")

    fig.suptitle("Autoencodeur — reconstruction", fontsize=13)
    plt.tight_layout()
    return fig, verdict_html


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

        with gr.Column(scale=2):
            image_out = gr.Image(label="Radio thoracique", type="numpy")
            label_out = gr.Markdown()

    gr.Markdown("---")

    with gr.Row():
        btn_eval_hybrid = gr.Button("Modèle hybride", variant="primary")

    gr.Markdown("---")

    infer_out = gr.Plot(label="Résultats — modèle hybride")

    gr.Markdown("---")

    with gr.Row():
        btn_eval_ae = gr.Button("Autoencodeur", variant="primary")

    gr.Markdown("---")

    ae_verdict = gr.HTML()
    ae_out = gr.Plot(label="Résultats — autoencodeur")

    index_slider.change(
        fn=show_image, inputs=index_slider, outputs=[image_out, label_out]
    )

    btn_eval_hybrid.click(fn=evaluate_hybrid, inputs=index_slider, outputs=infer_out)
    btn_eval_ae.click(fn=evaluate_ae, inputs=index_slider, outputs=[ae_out, ae_verdict])

    # Affichage initial
    demo.load(fn=show_image, inputs=index_slider, outputs=[image_out, label_out])


if __name__ == "__main__":
    demo.launch()
