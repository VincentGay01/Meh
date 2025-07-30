import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import torch.nn.functional as F
from pathlib import Path
import glob

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

def load_multiple_references(ref_paths, target_size):
    """Charge plusieurs images de référence"""
    references = []
    valid_paths = []
    
    for path in ref_paths:
        try:
            ref = load_rgb(path, target_size)
            references.append(ref)
            valid_paths.append(path)
            print(f"✅ Référence chargée: {path}")
        except Exception as e:
            print(f"⚠️  Impossible de charger {path}: {e}")
    
    if not references:
        raise ValueError("Aucune image de référence valide trouvée")
    
    return references, valid_paths

def estimate_flow_farneback_improved(img1, img2):
    """Version améliorée de Farneback avec réduction des artefacts de bloc"""
    img1_gray = cv2.cvtColor((img1.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor((img2.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Configurations optimisées pour éviter les artefacts de bloc
    configs = [
        # Config 1: Anti-blocs avec grandes fenêtres
        {'pyr_scale': 0.5, 'levels': 3, 'winsize': 21, 'iterations': 3, 'poly_n': 7, 'poly_sigma': 1.5},
        # Config 2: Lissage moyen avec overlap
        {'pyr_scale': 0.6, 'levels': 4, 'winsize': 31, 'iterations': 5, 'poly_n': 7, 'poly_sigma': 2.0},
        # Config 3: Très lisse pour structure globale
        {'pyr_scale': 0.7, 'levels': 5, 'winsize': 41, 'iterations': 8, 'poly_n': 9, 'poly_sigma': 2.5}
    ]
    
    flows = []
    for i, config in enumerate(configs):
        try:
            # Pré-lissage léger pour réduire le bruit haute fréquence
            img1_smooth = cv2.GaussianBlur(img1_gray, (3, 3), 0.8)
            img2_smooth = cv2.GaussianBlur(img2_gray, (3, 3), 0.8)
            
            flow = cv2.calcOpticalFlowFarneback(img1_smooth, img2_smooth, None, **config, flags=0)
            if flow is not None:
                flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
                # Post-lissage anti-artefacts
                flow_tensor = post_process_flow_anti_artifacts(flow_tensor)
                flows.append(flow_tensor)
                
        except Exception as e:
            print(f"    Erreur config {i+1}: {e}")
            continue
    
    if not flows:
        return torch.zeros(1, 2, img1.shape[2], img1.shape[3], device=device)
    
    # Moyenne pondérée privilégiant les configurations plus lisses
    weights = [0.2, 0.3, 0.5]  # Plus de poids aux configurations anti-artefacts
    flow_combined = sum(w * f for w, f in zip(weights[:len(flows)], flows))
    flow_combined = flow_combined / sum(weights[:len(flows)])
    
    return flow_combined

def post_process_flow_anti_artifacts(flow):
    """Post-traitement spécialisé anti-artefacts"""
    # 1. Détection et correction des sauts brusques
    flow = median_filter_flow(flow, kernel_size=3)
    
    # 2. Lissage directionnel sur les patterns réguliers
    flow = directional_smooth_flow(flow)
    
    # 3. Correction des discontinuités de bloc
    flow = remove_block_discontinuities(flow, block_size=8)
    
    return flow

def median_filter_flow(flow, kernel_size=3):
    """Filtre médian pour supprimer les outliers"""
    B, C, H, W = flow.shape
    flow_filtered = flow.clone()
    
    # Convertir en numpy pour utiliser le filtre médian de scipy
    for b in range(B):
        for c in range(C):
            flow_np = flow[b, c].cpu().numpy()
            try:
                from scipy.ndimage import median_filter
                flow_filtered_np = median_filter(flow_np, size=kernel_size)
                flow_filtered[b, c] = torch.from_numpy(flow_filtered_np).to(device)
            except ImportError:
                # Fallback: moyennage local
                kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size**2)
                flow_filtered[b:b+1, c:c+1] = F.conv2d(
                    flow[b:b+1, c:c+1], kernel, padding=kernel_size//2
                )
    
    return flow_filtered

def directional_smooth_flow(flow):
    """Lissage directionnel pour préserver les structures importantes"""
    # Noyaux directionnels pour détecter les patterns
    kernels = {
        'horizontal': torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]], device=device, dtype=torch.float32).view(1, 1, 3, 3) / 3,
        'vertical': torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], device=device, dtype=torch.float32).view(1, 1, 3, 3) / 3,
        'diagonal1': torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3) / 3,
        'diagonal2': torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=device, dtype=torch.float32).view(1, 1, 3, 3) / 3,
    }
    
    # Appliquer chaque noyau et prendre le minimum de variation
    flow_variants = []
    for direction, kernel in kernels.items():
        kernel_2ch = kernel.repeat(2, 1, 1, 1)
        flow_smooth = F.conv2d(flow, kernel_2ch, padding=1, groups=2)
        flow_variants.append(flow_smooth)
    
    # Choisir la variante avec le moins de variation locale
    flow_final = flow.clone()
    for variant in flow_variants:
        # Calculer la variation locale
        diff = torch.abs(variant - flow)
        variation = F.avg_pool2d(torch.sum(diff, dim=1, keepdim=True), 5, stride=1, padding=2)
        
        # Utiliser la variante là où elle produit moins de variation
        mask = variation < 0.1  # Seuil à ajuster
        flow_final = torch.where(mask, variant, flow_final)
    
    return flow_final

def compute_flow_confidence(flow1, flow2):
    """Calcule la confiance entre deux estimations de flux"""
    diff = torch.norm(flow1 - flow2, dim=1, keepdim=True)
    max_diff = torch.max(diff)
    if max_diff == 0:
        return torch.ones_like(diff)
    confidence = torch.exp(-diff / (max_diff * 0.5))
    return confidence

def estimate_multi_reference_flow(normals, references):
    """Estime le flux optique avec plusieurs références"""
    flows = []
    confidences = []
    
    print(f"Estimation du flux avec {len(references)} références...")
    
    for i, ref in enumerate(references):
        print(f"  Traitement référence {i+1}/{len(references)}")
        
        # Estimation avec Farneback amélioré
        flow_farneback = estimate_flow_farneback_improved(normals, ref)
        
        # Estimation avec RAFT si disponible
        try:
            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights).to(device).eval()
            with torch.no_grad():
                flow_list = model(normals, ref)
            flow_raft = flow_list[-1]
            
            # Combiner RAFT et Farneback
            confidence_raft_farneback = compute_flow_confidence(flow_raft, flow_farneback)
            flow_combined = confidence_raft_farneback * flow_raft + (1 - confidence_raft_farneback) * flow_farneback
            
        except Exception as e:
            print(f"    RAFT non disponible pour ref {i+1}, utilisation Farneback seul")
            flow_combined = flow_farneback
            confidence_raft_farneback = torch.ones_like(flow_farneback[:, :1])
        
        flows.append(flow_combined)
        confidences.append(confidence_raft_farneback)
    
    # Fusion intelligente des flux de toutes les références
    if len(flows) == 1:
        return flows[0]
    
    # Calculer les poids inter-références basés sur la cohérence
    inter_confidences = []
    for i in range(len(flows)):
        conf_sum = torch.zeros_like(confidences[i])
        for j in range(len(flows)):
            if i != j:
                conf_ij = compute_flow_confidence(flows[i], flows[j])
                conf_sum += conf_ij
        inter_confidences.append(conf_sum / (len(flows) - 1))
    
    # Fusion pondérée
    total_weight = torch.zeros_like(flows[0])
    final_flow = torch.zeros_like(flows[0])
    
    for flow, intra_conf, inter_conf in zip(flows, confidences, inter_confidences):
        weight = intra_conf * inter_conf
        final_flow += weight * flow
        total_weight += weight
    
    # Éviter la division par zéro
    total_weight = torch.clamp(total_weight, min=1e-8)
    final_flow = final_flow / total_weight
    
    # Lissage final adaptatif
    final_flow = adaptive_smooth_flow(final_flow, edge_threshold=1.5)
    
    return final_flow

def detect_block_artifacts(flow, block_size=8):
    """Détecte les artefacts de bloc dans le flux optique"""
    B, C, H, W = flow.shape
    
    # Créer une grille pour détecter les patterns de bloc
    artifact_score = torch.zeros(B, 1, H, W, device=device)
    
    # Vérifier les discontinuités aux frontières de blocs
    for y in range(block_size, H, block_size):
        for x in range(block_size, W, block_size):
            if y < H and x < W:
                # Mesurer la discontinuité à la frontière
                if y > 0:
                    diff_y = torch.abs(flow[:, :, y, :] - flow[:, :, y-1, :])
                    artifact_score[:, :, y-2:y+2, :] += torch.norm(diff_y, dim=1, keepdim=True).unsqueeze(2).repeat(1, 1, 4, 1)
                
                if x > 0:
                    diff_x = torch.abs(flow[:, :, :, x] - flow[:, :, :, x-1])
                    artifact_score[:, :, :, x-2:x+2] += torch.norm(diff_x, dim=1, keepdim=True).unsqueeze(3).repeat(1, 1, 1, 4)
    
    # Normaliser le score
    artifact_score = artifact_score / (artifact_score.max() + 1e-8)
    return artifact_score

def adaptive_smooth_flow(flow, edge_threshold=1.0):
    """Lissage adaptatif amélioré avec suppression des artefacts de bloc"""
    
    # 1. Détecter les artefacts de bloc
    block_artifacts = detect_block_artifacts(flow, block_size=8)
    
    # 2. Calculer les contours naturels
    flow_mag = torch.norm(flow, dim=1, keepdim=True)
    
    # Gradients avec Sobel pour plus de précision
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(flow_mag, sobel_x, padding=1)
    grad_y = F.conv2d(flow_mag, sobel_y, padding=1)
    
    natural_edges = torch.sqrt(grad_x**2 + grad_y**2)
    
    # 3. Distinguer contours naturels des artefacts
    # Les artefacts de bloc ont des patterns réguliers, les contours naturels sont plus organiques
    edge_strength = natural_edges * (1.0 - block_artifacts.squeeze(1).unsqueeze(1))
    
    # 4. Calculer les poids de lissage
    # Plus fort lissage sur les artefacts, préservation des contours naturels
    artifact_smooth_weight = torch.sigmoid(block_artifacts * 10 - 2)  # Fort lissage sur artefacts
    edge_preserve_weight = torch.exp(-edge_strength / edge_threshold)  # Préservation contours naturels
    
    # Combinaison: lissage fort sur artefacts, doux sur contours naturels
    smooth_weight = torch.max(artifact_smooth_weight, edge_preserve_weight)
    
    # 5. Lissage multi-échelle anti-artefacts
    kernel_configs = [
        {'size': 3, 'sigma': 0.8, 'weight': 0.4},   # Détails fins
        {'size': 7, 'sigma': 1.5, 'weight': 0.4},   # Structure moyenne
        {'size': 15, 'sigma': 3.0, 'weight': 0.2}   # Lissage global anti-blocs
    ]
    
    flow_final = flow.clone()
    
    for config in kernel_configs:
        k_size = config['size']
        sigma = config['sigma']
        base_weight = config['weight']
        
        # Créer noyau gaussien
        kernel_1d = torch.exp(-0.5 * (torch.arange(k_size, device=device, dtype=torch.float32) - k_size//2)**2 / sigma**2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.view(1, 1, k_size, 1) * kernel_1d.view(1, 1, 1, k_size)
        kernel_2d = kernel_2d.repeat(2, 1, 1, 1)
        
        flow_smooth_k = F.conv2d(flow, kernel_2d, padding=k_size//2, groups=2)
        
        # Poids adaptatif avec bonus anti-artefact
        adaptive_weight = smooth_weight * base_weight
        # Bonus de lissage sur les zones d'artefacts
        artifact_bonus = block_artifacts * 0.3
        final_weight = torch.clamp(adaptive_weight + artifact_bonus, 0, 1)
        
        flow_final = final_weight * flow_smooth_k + (1 - final_weight) * flow_final
    
    # 6. Post-traitement spécial anti-blocs
    flow_final = remove_block_discontinuities(flow_final)
    
    return flow_final

def remove_block_discontinuities(flow, block_size=8):
    """Supprime spécifiquement les discontinuités aux frontières de blocs"""
    B, C, H, W = flow.shape
    flow_corrected = flow.clone()
    
    # Noyau de lissage directionnel pour les frontières
    smooth_kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
    smooth_kernel = smooth_kernel.repeat(2, 1, 1, 1)
    
    # Traiter les frontières verticales des blocs
    for x in range(block_size, W, block_size):
        if x < W - 1:
            # Zone autour de la frontière
            start_x = max(0, x - 2)
            end_x = min(W, x + 3)
            
            # Appliquer un lissage local
            region = flow_corrected[:, :, :, start_x:end_x]
            region_smooth = F.conv2d(region, smooth_kernel, padding=1, groups=2)
            
            # Mélange progressif
            blend_weights = torch.linspace(0, 1, end_x - start_x, device=device).view(1, 1, 1, -1)
            blend_weights = torch.min(blend_weights, 1 - blend_weights) * 2  # Pic au centre
            
            flow_corrected[:, :, :, start_x:end_x] = (
                blend_weights * region_smooth + 
                (1 - blend_weights) * region
            )
    
    # Traiter les frontières horizontales des blocs
    for y in range(block_size, H, block_size):
        if y < H - 1:
            # Zone autour de la frontière
            start_y = max(0, y - 2)
            end_y = min(H, y + 3)
            
            # Appliquer un lissage local
            region = flow_corrected[:, :, start_y:end_y, :]
            region_smooth = F.conv2d(region, smooth_kernel, padding=1, groups=2)
            
            # Mélange progressif
            blend_weights = torch.linspace(0, 1, end_y - start_y, device=device).view(1, 1, -1, 1)
            blend_weights = torch.min(blend_weights, 1 - blend_weights) * 2  # Pic au centre
            
            flow_corrected[:, :, start_y:end_y, :] = (
                blend_weights * region_smooth + 
                (1 - blend_weights) * region
            )
    
    return flow_corrected

def warp(img, flow):
    """Warping avec gestion des bords améliorée"""
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
    
    return F.grid_sample(img, grid_sample, align_corners=True, mode='bilinear', padding_mode='reflection')

def compute_jacobian_robust(flow):
    """Calcul du jacobien plus robuste"""
    u, v = flow[:, 0:1], flow[:, 1:2]
    
    # Utiliser des différences finies centrées quand possible
    B, _, H, W = flow.shape
    
    # Gradients avec différences finies centrées
    du_dx = torch.zeros_like(u)
    du_dy = torch.zeros_like(u)
    dv_dx = torch.zeros_like(v)
    dv_dy = torch.zeros_like(v)
    
    # Différences finies centrées à l'intérieur
    du_dx[:, :, :, 1:-1] = (u[:, :, :, 2:] - u[:, :, :, :-2]) / 2.0
    du_dy[:, :, 1:-1, :] = (u[:, :, 2:, :] - u[:, :, :-2, :]) / 2.0
    dv_dx[:, :, :, 1:-1] = (v[:, :, :, 2:] - v[:, :, :, :-2]) / 2.0
    dv_dy[:, :, 1:-1, :] = (v[:, :, 2:, :] - v[:, :, :-2, :]) / 2.0
    
    # Différences avant/arrière sur les bords
    du_dx[:, :, :, 0] = u[:, :, :, 1] - u[:, :, :, 0]
    du_dx[:, :, :, -1] = u[:, :, :, -1] - u[:, :, :, -2]
    du_dy[:, :, 0, :] = u[:, :, 1, :] - u[:, :, 0, :]
    du_dy[:, :, -1, :] = u[:, :, -1, :] - u[:, :, -2, :]
    
    dv_dx[:, :, :, 0] = v[:, :, :, 1] - v[:, :, :, 0]
    dv_dx[:, :, :, -1] = v[:, :, :, -1] - v[:, :, :, -2]
    dv_dy[:, :, 0, :] = v[:, :, 1, :] - v[:, :, 0, :]
    dv_dy[:, :, -1, :] = v[:, :, -1, :] - v[:, :, -2, :]
    
    J = torch.zeros(B, H, W, 2, 2, device=device, dtype=flow.dtype)
    
    J[:, :, :, 0, 0] = 1.0 + du_dx.squeeze(1)
    J[:, :, :, 0, 1] = du_dy.squeeze(1)
    J[:, :, :, 1, 0] = dv_dx.squeeze(1)
    J[:, :, :, 1, 1] = 1.0 + dv_dy.squeeze(1)
    
    return J

def correct_normals_multi_ref(warped_normals, jacobian):
    """Correction des normales adaptée au multi-référence avec suppression des artefacts"""
    B, C, H, W = warped_normals.shape
    
    # Calculer le déterminant et conditionner
    det = torch.det(jacobian)
    cond_number = torch.norm(jacobian, dim=(-2, -1)) * torch.norm(torch.inverse(jacobian + torch.eye(2, device=device) * 1e-6), dim=(-2, -1))
    
    # Masque plus permissif pour le multi-référence
    reasonable_mask = (det > 0.1) & (det < 10.0) & (cond_number < 50.0) & torch.isfinite(det) & torch.isfinite(cond_number)
    
    # Conversion vers [-1,1]
    normals = warped_normals * 2.0 - 1.0
    
    # Correction des normales
    n_xy = normals[:, :2].permute(0, 2, 3, 1).unsqueeze(-1)
    
    # Utiliser pseudo-inverse pour plus de stabilité
    jacobian_reg = jacobian + torch.eye(2, device=device) * 1e-4
    try:
        corrected_xy = torch.matmul(jacobian_reg, n_xy).squeeze(-1)
    except:
        corrected_xy = n_xy.squeeze(-1)
    
    # Normalisation robuste
    xy_norm = torch.norm(corrected_xy, dim=-1, keepdim=True)
    xy_norm = torch.clamp(xy_norm, min=1e-8, max=10.0)
    corrected_xy = corrected_xy / xy_norm
    
    # Application du masque avec transition douce étendue
    original_xy = n_xy.squeeze(-1)
    mask_3d = reasonable_mask.unsqueeze(-1).float()
    
    # Transition douce avec noyau plus large pour éliminer les discontinuités
    smooth_kernel_size = 9  # Noyau plus grand
    smooth_kernel = torch.ones(1, 1, smooth_kernel_size, smooth_kernel_size, device=device) / (smooth_kernel_size**2)
    mask_smooth = F.conv2d(mask_3d.permute(0, 3, 1, 2), smooth_kernel, padding=smooth_kernel_size//2).permute(0, 2, 3, 1)
    
    # Application progressive avec suppression des frontières abruptes
    final_xy = mask_smooth * corrected_xy + (1 - mask_smooth) * original_xy
    
    # Post-lissage spécifique aux normales pour éliminer les artefacts restants
    final_xy = smooth_normal_transitions(final_xy.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    
    # Reconstruction finale
    corrected_normals = torch.cat([
        final_xy.permute(0, 3, 1, 2),
        normals[:, 2:3]
    ], dim=1)
    
    # Retour vers [0,1] avec clamping doux
    corrected_normals = (corrected_normals + 1.0) / 2.0
    corrected_normals = torch.clamp(corrected_normals, 0.0, 1.0)
    
    # Lissage final pour supprimer les derniers artefacts
    corrected_normals = final_normal_smoothing(corrected_normals)
    
    return corrected_normals

def smooth_normal_transitions(normals_xy):
    """Lissage spécifique des transitions de normales"""
    # Détecter les discontinuités dans les normales
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    
    discontinuity_map = torch.zeros_like(normals_xy[:, :1])
    
    for c in range(normals_xy.shape[1]):
        grad_x = F.conv2d(normals_xy[:, c:c+1], sobel_x, padding=1)
        grad_y = F.conv2d(normals_xy[:, c:c+1], sobel_y, padding=1)
        disc = torch.sqrt(grad_x**2 + grad_y**2)
        discontinuity_map += disc
    
    # Normaliser
    discontinuity_map = discontinuity_map / (discontinuity_map.max() + 1e-8)
    
    # Appliquer un lissage adaptatif
    smooth_strength = torch.sigmoid(discontinuity_map * 5 - 1)  # Fort lissage sur discontinuités
    
    # Lissage gaussien avec force variable
    kernel_sizes = [3, 5, 7]
    normals_smooth = normals_xy.clone()
    
    for k_size in kernel_sizes:
        sigma = k_size / 3.0
        kernel_1d = torch.exp(-0.5 * (torch.arange(k_size, device=device, dtype=torch.float32) - k_size//2)**2 / sigma**2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.view(1, 1, k_size, 1) * kernel_1d.view(1, 1, 1, k_size)
        
        for c in range(normals_xy.shape[1]):
            normals_c_smooth = F.conv2d(normals_xy[:, c:c+1], kernel_2d, padding=k_size//2)
            weight = smooth_strength * (0.3 if k_size == 3 else 0.4 if k_size == 5 else 0.3)
            normals_smooth[:, c:c+1] = weight * normals_c_smooth + (1 - weight) * normals_smooth[:, c:c+1]
    
    return normals_smooth

def final_normal_smoothing(normals):
    """Lissage final très doux pour éliminer les derniers artefacts"""
    # Lissage bilatéral approximé pour préserver les contours importants
    # tout en supprimant les artefacts de bloc
    
    # Calculer l'intensité pour guider le lissage
    intensity = torch.mean(normals, dim=1, keepdim=True)
    
    # Gradients de l'intensité
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(intensity, sobel_x, padding=1)
    grad_y = F.conv2d(intensity, sobel_y, padding=1)
    edge_strength = torch.sqrt(grad_x**2 + grad_y**2)
    
    # Poids de lissage inverse à la force des contours
    smooth_weight = torch.exp(-edge_strength * 3.0)
    
    # Lissage doux
    kernel = torch.ones(1, 1, 5, 5, device=device) / 25.0
    kernel = kernel.repeat(normals.shape[1], 1, 1, 1)
    
    normals_smooth = F.conv2d(normals, kernel, padding=2, groups=normals.shape[1])
    
    # Application du lissage avec préservation des contours
    smooth_weight = smooth_weight.repeat(1, normals.shape[1], 1, 1)
    normals_final = smooth_weight * normals_smooth + (1 - smooth_weight) * normals
    
    return normals_final


def save(img, path):
    """Sauvegarde avec gestion d'erreurs"""
    if img.dim() == 4:
        img = img.squeeze(0)
    
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255.0).astype(np.uint8)
    img_bgr = img_np[..., ::-1]
    
    # Créer le dossier si nécessaire
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    success = cv2.imwrite(path, img_bgr)
    if not success:
        raise RuntimeError(f"Impossible de sauvegarder l'image: {path}")

def align_normals_multi_ref(normal_path, ref_paths, out_path):
    """Fonction principale d'alignement multi-références"""
    try:
        # Validation des entrées
        if isinstance(ref_paths, str):
            # Si c'est un pattern ou un dossier
            if '*' in ref_paths or '?' in ref_paths:
                ref_paths = glob.glob(ref_paths)
            else:
                ref_paths = [ref_paths]
        
        if not ref_paths:
            raise ValueError("Aucune image de référence spécifiée")
        
        print(f"🔍 Recherche des références: {ref_paths}")
        
        # Charger la première référence pour déterminer la taille
        first_ref = cv2.imread(ref_paths[0])
        if first_ref is None:
            raise FileNotFoundError(f"Première référence introuvable: {ref_paths[0]}")
        
        target_size = (first_ref.shape[1], first_ref.shape[0])
        print(f"📏 Taille cible: {target_size}")
        
        # Charger les images
        normals = load_rgb(normal_path, target_size)
        references, valid_ref_paths = load_multiple_references(ref_paths, target_size)
        
        print(f"✅ Images chargées: 1 normale + {len(references)} références")
        
        # Estimation multi-référence du flux
        flow = estimate_multi_reference_flow(normals, references)
        
        print("🔄 Déformation des normales...")
        warped_normals = warp(normals, flow)
        
        print("📊 Calcul du jacobien...")
        jacobian = compute_jacobian_robust(flow)
        
        print("🔧 Correction des normales...")
        corrected_normals = correct_normals_multi_ref(warped_normals, jacobian)
        
        # Sauvegarde
        save(corrected_normals, out_path)
        print(f"✅ Résultat sauvegardé: {out_path}")
        
        # Statistiques
        flow_magnitude = torch.norm(flow, dim=1).mean().item()
        print(f"📈 Magnitude moyenne du flux: {flow_magnitude:.3f}")
        
        return corrected_normals
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise

# Fonction de commodité pour traitement par lot
def batch_align_normals(normal_dir, ref_dir, output_dir, pattern="*.png"):
    """Traite plusieurs normales avec les mêmes références"""
    normal_files = glob.glob(str(Path(normal_dir) / pattern))
    ref_files = glob.glob(str(Path(ref_dir) / pattern))
    
    if not normal_files:
        print(f"❌ Aucun fichier normal trouvé dans {normal_dir}")
        return
    
    if not ref_files:
        print(f"❌ Aucun fichier référence trouvé dans {ref_dir}")
        return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Traitement par lot: {len(normal_files)} normales avec {len(ref_files)} références")
    
    for normal_file in normal_files:
        try:
            normal_name = Path(normal_file).stem
            output_file = Path(output_dir) / f"{normal_name}_aligned.png"
            
            print(f"\n📝 Traitement: {normal_name}")
            align_normals_multi_ref(normal_file, ref_files, str(output_file))
            
        except Exception as e:
            print(f"❌ Erreur avec {normal_file}: {e}")

def get_all_images_in_folder(folder_path, extensions=None):
    """Récupère toutes les images dans un dossier"""
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable: {folder_path}")
    
    image_files = []
    for ext in extensions:
        image_files.extend(folder.glob(f"*{ext}"))
        image_files.extend(folder.glob(f"*{ext.upper()}"))
    
    image_files = [str(f) for f in image_files]
    
    if not image_files:
        raise ValueError(f"Aucune image trouvée dans {folder_path}")
    
    print(f"🖼️  Trouvé {len(image_files)} images dans {folder_path}")
    for img in sorted(image_files):
        print(f"    - {Path(img).name}")
    
    return sorted(image_files)

if __name__ == "__main__":
    # Utilisation avec toutes les images du dossier D:/project/meh/aligned_result
    reference_folder = "D:/project/meh/aligned_results"
    
    try:
        # Récupérer toutes les images du dossier
        all_references = get_all_images_in_folder(reference_folder)
        
        # Alignement avec toutes les références trouvées
        align_normals_multi_ref(
            "normals.png",  # Votre image de normales
            all_references,  # Toutes les images du dossier
            "normales_alignées_avec_toutes_refs.png"
        )
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement: {e}")
        print("\n💡 Assurez-vous que:")
        print("   - Le dossier D:/project/meh/aligned_result existe")
        print("   - Il contient des images (png, jpg, etc.)")
        print("   - Le fichier normals.png existe dans le dossier courant")