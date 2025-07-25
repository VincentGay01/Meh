import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_rgb(path, size=None):
    """Charge une image RGB et la convertit en tensor PyTorch"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {path}")
    
    img = img[..., ::-1]  # BGR → RGB
    if size: 
        img = cv2.resize(img, size)
    
    t = torch.from_numpy(img).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).to(device)

def estimate_flow_farneback(img1, img2):
    """Utilise l'algorithme de Farneback (plus lisse que RAFT)"""
    # Convertir en niveaux de gris
    img1_gray = cv2.cvtColor((img1.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor((img2.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Calculer le flux optique avec Farneback (plus lisse)
    flow = cv2.calcOpticalFlowPyrLK_dense(img1_gray, img2_gray)
    
    # Si ça ne marche pas, utiliser Farneback
    if flow is None:
        flow = cv2.calcOpticalFlowFarneback(
            img1_gray, img2_gray, None,
            pyr_scale=0.5,      # Échelle pyramidale
            levels=5,           # Nombre de niveaux
            winsize=13,         # Taille de fenêtre
            iterations=10,      # Itérations par niveau
            poly_n=5,          # Voisinage polynomial
            poly_sigma=1.1,    # Écart-type gaussien
            flags=0
        )
    
    # Convertir en tensor PyTorch
    flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return flow_tensor

def estimate_flow_lucas_kanade(img1, img2):
    """Utilise Lucas-Kanade dense pour un flux plus lisse"""
    img1_gray = cv2.cvtColor((img1.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor((img2.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Paramètres Lucas-Kanade
    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Créer une grille de points
    h, w = img1_gray.shape
    y, x = np.mgrid[0:h:8, 0:w:8].reshape(2, -1).astype(np.float32)
    points = np.column_stack((x, y))
    
    # Calculer le flux optique
    next_points, status, error = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, points, None, **lk_params)
    
    # Créer un champ de flux dense par interpolation
    good_points = points[status.flatten() == 1]
    good_next = next_points[status.flatten() == 1]
    flow_vectors = good_next - good_points
    
    # Interpolation pour créer un champ dense
    from scipy.interpolate import griddata
    
    xi, yi = np.mgrid[0:w, 0:h]
    flow_x = griddata(good_points, flow_vectors[:, 0], (xi, yi), method='cubic', fill_value=0)
    flow_y = griddata(good_points, flow_vectors[:, 1], (xi, yi), method='cubic', fill_value=0)
    
    flow = np.stack([flow_x.T, flow_y.T], axis=2)
    flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    return flow_tensor

def estimate_flow_hybrid(img1, img2):
    """Combinaison RAFT + méthodes classiques pour réduire les artefacts"""
    # Méthode 1: RAFT (précis mais avec artefacts)
    try:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights).to(device).eval()
        with torch.no_grad():
            flow_list = model(img1, img2)
        flow_raft = flow_list[-1]
    except:
        flow_raft = torch.zeros(1, 2, img1.shape[2], img1.shape[3], device=device)
    
    # Méthode 2: Farneback (plus lisse)
    try:
        flow_farneback = estimate_flow_farneback(img1, img2)
    except:
        flow_farneback = torch.zeros_like(flow_raft)
    
    # Combiner les deux approches
    # RAFT pour les détails fins, Farneback pour la structure globale
    alpha = 0.3  # Poids pour RAFT
    flow_combined = alpha * flow_raft + (1 - alpha) * flow_farneback
    
    # Lissage final avec un noyau adaptatif
    flow_smooth = adaptive_smooth_flow(flow_combined)
    
    return flow_smooth

def adaptive_smooth_flow(flow, edge_threshold=1.0):
    """Lissage adaptatif qui préserve les contours importants"""
    # Calculer la magnitude du gradient pour détecter les contours
    flow_mag = torch.norm(flow, dim=1, keepdim=True)
    
    # Calculer les gradients de la magnitude
    grad_x = flow_mag[:, :, :, 1:] - flow_mag[:, :, :, :-1]
    grad_y = flow_mag[:, :, 1:, :] - flow_mag[:, :, :-1, :]
    
    # Pad pour maintenir la taille
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
    
    # Calculer la force du contour
    edge_strength = torch.sqrt(grad_x**2 + grad_y**2)
    
    # Créer un masque de lissage (plus fort là où il n'y a pas de contours)
    smooth_weight = torch.exp(-edge_strength / edge_threshold)
    
    # Appliquer un lissage gaussien pondéré
    kernel_sizes = [3, 5, 7]
    flows_smooth = []
    
    for k_size in kernel_sizes:
        sigma = k_size / 6.0
        kernel_1d = torch.exp(-0.5 * (torch.arange(k_size, device=device, dtype=torch.float32) - k_size//2)**2 / sigma**2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.view(1, 1, k_size, 1) * kernel_1d.view(1, 1, 1, k_size)
        kernel_2d = kernel_2d.repeat(2, 1, 1, 1)
        
        flow_smooth_k = F.conv2d(flow, kernel_2d, padding=k_size//2, groups=2)
        flows_smooth.append(flow_smooth_k)
    
    # Combiner les différents niveaux de lissage selon la force du contour
    flow_final = flow.clone()
    for i, flow_smooth_k in enumerate(flows_smooth):
        weight = smooth_weight * (0.5 ** i)  # Poids décroissants
        flow_final = weight * flow_smooth_k + (1 - weight) * flow_final
    
    return flow_final

def warp(img, flow):
    """Warping standard"""
    B, C, H, W = img.shape
    
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32), 
        torch.arange(W, device=device, dtype=torch.float32), 
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)
    
    new_grid = grid + flow
    new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0
    new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0
    
    grid_sample = new_grid.permute(0, 2, 3, 1)
    
    return F.grid_sample(img, grid_sample, align_corners=True, mode='bilinear', padding_mode='border')

def compute_jacobian_simple(flow):
    """Jacobien simple"""
    u, v = flow[:, 0:1], flow[:, 1:2]
    
    du_dx = torch.zeros_like(u)
    du_dy = torch.zeros_like(u)
    dv_dx = torch.zeros_like(v)
    dv_dy = torch.zeros_like(v)
    
    du_dx[:, :, :, :-1] = u[:, :, :, 1:] - u[:, :, :, :-1]
    du_dy[:, :, :-1, :] = u[:, :, 1:, :] - u[:, :, :-1, :]
    dv_dx[:, :, :, :-1] = v[:, :, :, 1:] - v[:, :, :, :-1]
    dv_dy[:, :, :-1, :] = v[:, :, 1:, :] - v[:, :, :-1, :]
    
    B, _, H, W = flow.shape
    J = torch.zeros(B, H, W, 2, 2, device=device, dtype=flow.dtype)
    
    J[:, :, :, 0, 0] = 1.0 + du_dx.squeeze(1)
    J[:, :, :, 0, 1] = du_dy.squeeze(1)
    J[:, :, :, 1, 0] = dv_dx.squeeze(1)
    J[:, :, :, 1, 1] = 1.0 + dv_dy.squeeze(1)
    
    return J

def correct_normals_conservative(warped_normals, jacobian):
    """Correction conservative qui minimise les artefacts"""
    B, C, H, W = warped_normals.shape
    
    # Calculer le déterminant du jacobien pour détecter les déformations extrêmes
    det = torch.det(jacobian)  # [B,H,W]
    
    # Masque pour les transformations raisonnables
    reasonable_mask = (det > 0.2) & (det < 5.0) & (torch.abs(det - 1.0) < 2.0)
    
    # Convertir vers [-1,1]
    normals = warped_normals * 2.0 - 1.0
    
    # Pour les régions avec transformations raisonnables, appliquer la correction
    n_xy = normals[:, :2].permute(0, 2, 3, 1).unsqueeze(-1)  # [B,H,W,2,1]
    corrected_xy = torch.matmul(jacobian, n_xy).squeeze(-1)  # [B,H,W,2]
    
    # Normaliser
    xy_norm = torch.norm(corrected_xy, dim=-1, keepdim=True) + 1e-8
    corrected_xy = corrected_xy / xy_norm
    
    # Appliquer seulement dans les régions raisonnables
    original_xy = n_xy.squeeze(-1)
    mask_3d = reasonable_mask.unsqueeze(-1).float()
    final_xy = mask_3d * corrected_xy + (1 - mask_3d) * original_xy
    
    # Garder Z original
    corrected_normals = torch.cat([
        final_xy.permute(0, 3, 1, 2),  # [B,2,H,W]
        normals[:, 2:3]  # [B,1,H,W]
    ], dim=1)
    
    # Retour vers [0,1]
    corrected_normals = (corrected_normals + 1.0) / 2.0
    
    return torch.clamp(corrected_normals, 0.0, 1.0)

def save(img, path):
    """Sauvegarde"""
    if img.dim() == 4:
        img = img.squeeze(0)
    
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255.0).astype(np.uint8)
    img_bgr = img_np[..., ::-1]
    
    success = cv2.imwrite(path, img_bgr)
    if not success:
        raise RuntimeError(f"Impossible de sauvegarder l'image: {path}")

def align_normals_raft(normal_path, ref_path, out_path, method='hybrid'):
    """Fonction principale avec choix de méthode"""
    try:
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            raise FileNotFoundError(f"Image de référence introuvable: {ref_path}")
        
        size = (ref_img.shape[1], ref_img.shape[0])
        print(f"Taille cible: {size}")
        
        normals = load_rgb(normal_path, size)
        ref_tensor = load_rgb(ref_path, size)
        
        print("Images chargées avec succès")
        
        # Choisir la méthode d'estimation du flux
        if method == 'hybrid':
            print("Estimation hybride du flux optique...")
            flow = estimate_flow_hybrid(normals, ref_tensor)
        elif method == 'farneback':
            print("Estimation Farneback du flux optique...")
            flow = estimate_flow_farneback(normals, ref_tensor)
        else:  # raft
            print("Estimation RAFT du flux optique...")
            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights).to(device).eval()
            with torch.no_grad():
                flow_list = model(normals, ref_tensor)
            flow = flow_list[-1]
        
        print("Déformation des normales...")
        warped_normals = warp(normals, flow)
        
        print("Calcul du jacobien...")
        jacobian = compute_jacobian_simple(flow)
        
        print("Correction conservative des normales...")
        corrected_normals = correct_normals_conservative(warped_normals, jacobian)
        
        save(corrected_normals, out_path)
        print(f"✅ Image corrigée sauvegardée: {out_path}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise

if __name__ == "__main__":
    # Essayer la méthode hybride d'abord
    align_normals_raft("normals.png", "404.png", "normales_alignées_raft.png", method='hybrid')