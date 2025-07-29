from PIL import Image
import torchvision.transforms as T
import torch
import os
import spectral
from RTIFeatureNet import RTIFeatureNet
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from HSIFeatureNet import HSIFeatureNet
import numpy as np
import json
from datetime import datetime

# Créer un dossier principal pour tous les résultats
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_output_folder = f"D:/project/meh/results_{timestamp}/"
os.makedirs(main_output_folder, exist_ok=True)

# Paramètres
rti_folder = "D:/project/meh/aligned_results/"
N = 16  # nombre d'images RTI
H, W = 512, 512  # taille des images

# Transformations
to_tensor = T.Compose([
    T.Resize((H, W)),
    T.ToTensor(),  # convertit en (C, H, W) avec valeurs ∈ [0, 1]
])

# Charger les RTI
rti_images = []
for i in range(N):
    img_path = os.path.join(rti_folder, f"rti_{i}.jpg")
    img = Image.open(img_path).convert("L")  # L = niveau de gris
    img_tensor = to_tensor(img)  # shape = (1, H, W)
    rti_images.append(img_tensor)

rti_stack = torch.cat(rti_images, dim=0)  # shape = (N, H, W)
rti_stack = rti_stack.unsqueeze(0)         # → (1, N, H, W)

# ==================== RTI FEATURE NET ====================
print("Traitement RTIFeatureNet...")
model = RTIFeatureNet(num_rti_views=N, out_features=64)
output_features = model(rti_stack)  # shape = (1, 64, H, W)

# Supprimer la dimension batch : (64, H, W)
features = output_features.squeeze(0).detach().cpu()

# SAUVEGARDE 1 : Features RTI en format numpy
rti_folder_output = os.path.join(main_output_folder, "rti_features/")
os.makedirs(rti_folder_output, exist_ok=True)

# Sauver les features brutes en .npy (format numpy)
np.save(os.path.join(rti_folder_output, "rti_features_raw.npy"), features.numpy())
print(f"✓ Features RTI sauvées : {rti_folder_output}rti_features_raw.npy")

# SAUVEGARDE 2 : Features RTI individuelles en images PNG
for i, fmap in enumerate(features):
    # Normaliser entre 0-255
    fmap_norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
    fmap_img = (fmap_norm * 255).byte().numpy()
    img = Image.fromarray(fmap_img)
    img.save(os.path.join(rti_folder_output, f"feature_{i:02d}.png"))

print(f"✓ {len(features)} cartes de features RTI sauvées en PNG")

# SAUVEGARDE 3 : Carte de saillance RTI
saliency_map_rti = torch.norm(features, dim=0)  # sqrt(sum(features**2, dim=0))
saliency_map_rti_norm = (saliency_map_rti - saliency_map_rti.min()) / (saliency_map_rti.max() - saliency_map_rti.min() + 1e-8)

# Sauver en numpy
np.save(os.path.join(rti_folder_output, "saliency_map_rti.npy"), saliency_map_rti.numpy())

# Sauver en image
saliency_img = (saliency_map_rti_norm * 255).byte().numpy()
Image.fromarray(saliency_img).save(os.path.join(rti_folder_output, "saliency_map_rti.png"))
print("✓ Carte de saillance RTI sauvée")

# Visualisation RTI
nb_to_show = 8
plt.figure(figsize=(15, 5))
for i in range(nb_to_show):
    plt.subplot(1, nb_to_show, i + 1)
    fmap = features[i]
    plt.imshow(fmap, cmap="viridis")
    plt.axis("off")
    plt.title(f"Feature {i}")
plt.suptitle("Cartes de feature RTIFeatureNet")
plt.tight_layout()
plt.savefig(os.path.join(rti_folder_output, "rti_features_visualization.png"), dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(saliency_map_rti_norm.cpu().numpy(), cmap="hot")
plt.colorbar()
plt.title("Carte de saillance RTI (norme L2 des features)")
plt.axis("off")
plt.savefig(os.path.join(rti_folder_output, "rti_saliency_visualization.png"), dpi=300, bbox_inches='tight')
plt.show()

# ==================== HSI FEATURE NET ====================
print("Traitement HSIFeatureNet...")
hdr_path = "D:/project/meh/capture/404.hdr"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement de l'image hyperspectrale
hsi = spectral.open_image(hdr_path)
hsi_np = hsi.load().astype("float32")  # shape: (H, W, C)
hsi_np = hsi_np.transpose((2, 0, 1))  # → (C, H, W)
hsi_tensor = torch.from_numpy(hsi_np).unsqueeze(0).to(device)  # → (1, C, H, W)

# Lancer le modèle HSI
num_bands = hsi_tensor.shape[1]
modeldeux = HSIFeatureNet(num_spectral_bands=num_bands, out_features=64).to(device)
modeldeux.eval()

with torch.no_grad():
    features_hsi, saliency_map_hsi = modeldeux(hsi_tensor)

# SAUVEGARDE 4 : Features HSI
hsi_folder_output = os.path.join(main_output_folder, "hsi_features/")
os.makedirs(hsi_folder_output, exist_ok=True)

# Sauver les features HSI brutes
features_hsi_cpu = features_hsi[0].cpu().detach().numpy()  # (D, H, W)
np.save(os.path.join(hsi_folder_output, "hsi_features_raw.npy"), features_hsi_cpu)
print(f"✓ Features HSI sauvées : {hsi_folder_output}hsi_features_raw.npy")

# SAUVEGARDE 5 : Features HSI individuelles en images
for i in range(min(features_hsi_cpu.shape[0], 32)):  # limiter à 32 features max
    fmap = features_hsi_cpu[i]
    fmap_norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
    fmap_img = (fmap_norm * 255).astype(np.uint8)
    Image.fromarray(fmap_img).save(os.path.join(hsi_folder_output, f"hsi_feature_{i:02d}.png"))

print(f"✓ Features HSI individuelles sauvées en PNG")

# SAUVEGARDE 6 : Carte de saillance HSI
saliency_hsi = saliency_map_hsi.squeeze().cpu().numpy()
np.save(os.path.join(hsi_folder_output, "saliency_map_hsi.npy"), saliency_hsi)

# Normaliser et sauver en image
saliency_hsi_norm = (saliency_hsi - saliency_hsi.min()) / (saliency_hsi.max() - saliency_hsi.min() + 1e-8)
saliency_hsi_img = (saliency_hsi_norm * 255).astype(np.uint8)
Image.fromarray(saliency_hsi_img).save(os.path.join(hsi_folder_output, "saliency_map_hsi.png"))
print("✓ Carte de saillance HSI sauvée")

# Visualisation HSI
plt.figure(figsize=(8, 6))
plt.imshow(saliency_hsi, cmap='hot')
plt.title("Carte de saillance HSI")
plt.axis("off")
plt.colorbar()
plt.savefig(os.path.join(hsi_folder_output, "hsi_saliency_visualization.png"), dpi=300, bbox_inches='tight')
plt.show()

num_maps = min(features_hsi_cpu.shape[0], 16)
plt.figure(figsize=(12, 8))
for i in range(num_maps):
    plt.subplot(4, 4, i+1)
    plt.imshow(features_hsi_cpu[i], cmap='viridis')
    plt.title(f"Feature {i}")
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(hsi_folder_output, "hsi_features_visualization.png"), dpi=300, bbox_inches='tight')
plt.show()

# ==================== SAUVEGARDE MÉTADONNÉES ====================
metadata = {
    "timestamp": timestamp,
    "rti_params": {
        "num_images": N,
        "image_size": [H, W],
        "num_features": features.shape[0],
        "source_folder": rti_folder
    },
    "hsi_params": {
        "source_file": hdr_path,
        "num_bands": num_bands,
        "num_features": features_hsi_cpu.shape[0],
        "image_shape": list(features_hsi_cpu.shape[1:])
    },
    "device": str(device),
    "files_created": {
        "rti_features_raw": "rti_features/rti_features_raw.npy",
        "rti_saliency": "rti_features/saliency_map_rti.npy",
        "hsi_features_raw": "hsi_features/hsi_features_raw.npy",
        "hsi_saliency": "hsi_features/saliency_map_hsi.npy"
    }
}

with open(os.path.join(main_output_folder, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*50)
print(f" TOUS LES RÉSULTATS SAUVÉS DANS : {main_output_folder}")
print("="*50)
print("Structure des fichiers :")
print("├── rti_features/")
print("│   ├── rti_features_raw.npy          # Features brutes")
print("│   ├── saliency_map_rti.npy          # Carte saillance")
print("│   ├── feature_XX.png                # Features individuelles")
print("│   └── *.png                         # Visualisations")
print("├── hsi_features/")
print("│   ├── hsi_features_raw.npy          # Features brutes")
print("│   ├── saliency_map_hsi.npy          # Carte saillance")
print("│   ├── hsi_feature_XX.png            # Features individuelles")
print("│   └── *.png                         # Visualisations")
print("└── metadata.json                     # Infos sur l'expérience")

# ==================== FONCTION DE RECHARGEMENT ====================
def load_results(results_folder):
    """
    Fonction pour recharger les résultats sauvés
    """
    # Charger métadonnées
    with open(os.path.join(results_folder, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    # Charger features RTI
    rti_features = np.load(os.path.join(results_folder, "rti_features/rti_features_raw.npy"))
    rti_saliency = np.load(os.path.join(results_folder, "rti_features/saliency_map_rti.npy"))
    
    # Charger features HSI
    hsi_features = np.load(os.path.join(results_folder, "hsi_features/hsi_features_raw.npy"))
    hsi_saliency = np.load(os.path.join(results_folder, "hsi_features/saliency_map_hsi.npy"))
    
    return {
        "metadata": metadata,
        "rti_features": rti_features,
        "rti_saliency": rti_saliency,
        "hsi_features": hsi_features,
        "hsi_saliency": hsi_saliency
    }

print(f"\n💡 Pour recharger ces résultats plus tard :")
print(f"results = load_results('{main_output_folder}')")