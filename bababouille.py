#bababouille
from PIL import Image
import torchvision.transforms as T
import torch
import os
import spectral
from RTIFeatureNet import RTIFeatureNet
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from HSIFeatureNet import HSIFeatureNet
# Paramètres
rti_folder = "D:/project/meh/aligned_results/"
N = 16  # nombre d’images RTI
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
model = RTIFeatureNet(num_rti_views=N, out_features=64)
output_features = model(rti_stack)  # shape = (1, 64, H, W)


# Supprimer la dimension batch : (64, H, W)
features = output_features.squeeze(0).detach().cpu()

# Nombre de cartes à afficher
nb_to_show = 8  # tu peux changer ça

plt.figure(figsize=(15, 5))
for i in range(nb_to_show):
    plt.subplot(1, nb_to_show, i + 1)
    fmap = features[i]
    plt.imshow(fmap, cmap="viridis")
    plt.axis("off")
    plt.title(f"Feature {i}")
plt.suptitle("Cartes de feature RTIFeatureNet")
plt.tight_layout()
plt.show()


save_folder = "D:/project/meh/features_output/"
os.makedirs(save_folder, exist_ok=True)

for i, fmap in enumerate(features):
    # Normaliser entre 0-255
    fmap_norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
    fmap_img = (fmap_norm * 255).byte().numpy()
    img = Image.fromarray(fmap_img)
    img.save(os.path.join(save_folder, f"feature_{i:02d}.png"))
    
    
    
# Norme L2 par pixel → carte de saillance (H, W)
saliency_map = torch.norm(features, dim=0)  # sqrt(sum(features**2, dim=0))

# Normaliser pour affichage
saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

plt.imshow(saliency_map.cpu().numpy(), cmap="hot")
plt.colorbar()
plt.title("Carte de saillance (norme L2 des features)")
plt.axis("off")
plt.show()



# --- PARAMÈTRES ---
hdr_path = "D:/project/meh/capture/404.hdr"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ÉTAPE 1 : Chargement de l'image hyperspectrale ---
hsi = spectral.open_image(hdr_path)  # charge .hdr et .img/.raw automatiquement
hsi_np = hsi.load().astype("float32")  # shape: (H, W, C)

# --- ÉTAPE 2 : Passage en tenseur PyTorch ---
hsi_np = hsi_np.transpose((2, 0, 1))  # → (C, H, W)
hsi_tensor = torch.from_numpy(hsi_np).unsqueeze(0).to(device)  # → (1, C, H, W)

# --- ÉTAPE 3 : Lancer le modèle ---
num_bands = hsi_tensor.shape[1]
modeldeux= HSIFeatureNet(num_spectral_bands=num_bands, out_features=64).to(device)
modeldeux.eval()

with torch.no_grad():
    features, saliency_map = modeldeux(hsi_tensor)

# --- ÉTAPE 4 : Visualiser la carte de saillance ---
saliency = saliency_map.squeeze().cpu().numpy()
plt.imshow(saliency, cmap='hot')
plt.title("Carte de saillance HSI")
plt.axis("off")
plt.colorbar()
plt.show()



features_np = features[0].cpu().detach().numpy()  # (D, H, W)
num_maps = min(features_np.shape[0], 16)  # nombre de features à afficher

plt.figure(figsize=(12, 8))
for i in range(num_maps):
    plt.subplot(4, 4, i+1)
    plt.imshow(features_np[i], cmap='viridis')
    plt.title(f"Feature {i}")
    plt.axis("off")
plt.tight_layout()
plt.show()