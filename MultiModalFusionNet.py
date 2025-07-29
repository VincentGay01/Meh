import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import json
import os
from datetime import datetime
from PIL import Image
import cv2

class MultiModalFusionNet(nn.Module):
    """
    Réseau de fusion multimodale pour détection d'anomalies
    Combine les features RTI et HSI pour détecter les anomalies
    """
    def __init__(self, rti_features=64, hsi_features=64, fusion_dim=128):
        super(MultiModalFusionNet, self).__init__()
        
        # Encodeurs pour chaque modalité
        self.rti_encoder = nn.Sequential(
            nn.Conv2d(rti_features, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.hsi_encoder = nn.Sequential(
            nn.Conv2d(hsi_features, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Module de fusion avec attention
        self.attention_rti = nn.Sequential(
            nn.Conv2d(16, 8, 1),
            nn.Sigmoid()
        )
        
        self.attention_hsi = nn.Sequential(
            nn.Conv2d(16, 8, 1),
            nn.Sigmoid()
        )
        
        # Fusion des modalités
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(32, fusion_dim, 3, padding=1),  # 16+16 = 32 channels
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(),
            nn.Conv2d(fusion_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Décodeur pour reconstruction (autoencoder)
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),  # Sortie 1 channel pour anomaly score
            nn.Sigmoid()
        )
        
        # Tête de classification pour anomalie
        self.anomaly_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),  # Score d'anomalie par pixel
            nn.Sigmoid()
        )
    
    def forward(self, rti_features, hsi_features):
        # Encodage de chaque modalité
        rti_encoded = self.rti_encoder(rti_features)  # (B, 16, H, W)
        hsi_encoded = self.hsi_encoder(hsi_features)  # (B, 16, H, W)
        
        # Mécanisme d'attention
        rti_attention = self.attention_rti(rti_encoded)  # (B, 8, H, W)
        hsi_attention = self.attention_hsi(hsi_encoded)  # (B, 8, H, W)
        
        # Application de l'attention
        rti_weighted = rti_encoded * rti_attention.repeat(1, 2, 1, 1)  # (B, 16, H, W)
        hsi_weighted = hsi_encoded * hsi_attention.repeat(1, 2, 1, 1)  # (B, 16, H, W)
        
        # Fusion des modalités
        fused = torch.cat([rti_weighted, hsi_weighted], dim=1)  # (B, 32, H, W)
        fused_features = self.fusion_conv(fused)  # (B, 32, H, W)
        
        # Reconstruction et détection d'anomalie
        reconstructed = self.decoder(fused_features)  # (B, 1, H, W)
        anomaly_score = self.anomaly_head(fused_features)  # (B, 1, H, W)
        
        return {
            'anomaly_score': anomaly_score,
            'reconstructed': reconstructed,
            'fused_features': fused_features,
            'rti_attention': rti_attention,
            'hsi_attention': hsi_attention
        }

class AnomalyDetectionPipeline:
    """
    Pipeline complet pour la détection d'anomalies multimodale
    """
    def __init__(self, device='cpu'):
        # Forcer CPU si CUDA n'est pas disponible
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA non disponible, utilisation du CPU")
            device = 'cpu'
        
        self.device = torch.device(device)
        self.model = None
        self.scaler_rti = StandardScaler()
        self.scaler_hsi = StandardScaler()
        
        print(f"🖥️  Device utilisé: {self.device}")
        
    def load_features(self, results_folder):
        """Charger les features RTI et HSI depuis les résultats sauvés"""
        # Charger métadonnées
        with open(os.path.join(results_folder, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)
        
        # Charger features
        rti_features = np.load(os.path.join(results_folder, "rti_features/rti_features_raw.npy"))
        hsi_features = np.load(os.path.join(results_folder, "hsi_features/hsi_features_raw.npy"))
        
        # Convertir en tenseurs PyTorch
        self.rti_features = torch.from_numpy(rti_features).float().unsqueeze(0).to(self.device)
        self.hsi_features = torch.from_numpy(hsi_features).float().unsqueeze(0).to(self.device)
        
        print(f"✓ Features chargées - RTI: {self.rti_features.shape}, HSI: {self.hsi_features.shape}")
        return self.rti_features, self.hsi_features
    
    def initialize_model(self, rti_dim=64, hsi_dim=64):
        """Initialiser le modèle de fusion"""
        self.model = MultiModalFusionNet(
            rti_features=rti_dim, 
            hsi_features=hsi_dim
        ).to(self.device)
        
        print(f"✓ Modèle initialisé sur {self.device}")
        return self.model
    
    def create_synthetic_anomalies(self, rti_features, hsi_features, num_anomalies=5):
        """
        Créer des anomalies synthétiques pour l'entraînement non-supervisé
        """
        H, W = rti_features.shape[-2:]
        anomaly_masks = []
        
        for _ in range(num_anomalies):
            # Anomalie circulaire aléatoire
            mask = np.zeros((H, W), dtype=np.float32)
            center_x, center_y = np.random.randint(50, W-50), np.random.randint(50, H-50)
            radius = np.random.randint(10, 30)
            
            y, x = np.ogrid[:H, :W]
            anomaly_region = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            mask[anomaly_region] = 1.0
            
            anomaly_masks.append(torch.from_numpy(mask).to(self.device))
        
        return anomaly_masks
    
    def train_unsupervised(self, rti_features, hsi_features, epochs=50, lr=0.001):
        """
        Entraînement non-supervisé du modèle (optimisé pour CPU)
        """
        if self.model is None:
            self.initialize_model(rti_features.shape[1], hsi_features.shape[1])
        
        # Optimiseur avec paramètres adaptés au CPU
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Perte combinée : reconstruction + anomalie
        mse_loss = nn.MSELoss()
        
        self.model.train()
        losses = []
        
        print(f"Début de l'entraînement non-supervisé sur {self.device}...")
        print(f"Epochs réduits à {epochs} pour optimiser le temps CPU")
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(rti_features, hsi_features)
            
            # Perte de reconstruction (autoencoder)
            # On utilise la moyenne des features comme cible de reconstruction
            target_reconstruction = torch.mean(torch.cat([rti_features, hsi_features], dim=1), dim=1, keepdim=True)
            target_reconstruction = F.interpolate(target_reconstruction, size=outputs['reconstructed'].shape[-2:])
            
            loss_reconstruction = mse_loss(outputs['reconstructed'], target_reconstruction)
            
            # Perte de régularisation pour forcer la sparsité des anomalies
            anomaly_sparsity = torch.mean(outputs['anomaly_score'])
            
            # Perte de cohérence entre modalités (réduite pour CPU)
            rti_attention_loss = torch.var(outputs['rti_attention'])
            hsi_attention_loss = torch.var(outputs['hsi_attention'])
            
            # Perte totale (coefficients ajustés pour CPU)
            total_loss = (loss_reconstruction + 
                         0.05 * anomaly_sparsity + 
                         0.005 * (rti_attention_loss + hsi_attention_loss))
            
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {total_loss.item():.6f}")
        
        print("✓ Entraînement terminé")
        return losses
    
    def detect_anomalies(self, rti_features, hsi_features, threshold=0.5):
        """
        Détecter les anomalies sur une image
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(rti_features, hsi_features)
        
        # Score d'anomalie par pixel
        anomaly_map = outputs['anomaly_score'].squeeze().cpu().numpy()
        
        # Créer le masque binaire d'anomalie
        anomaly_mask = (anomaly_map > threshold).astype(np.uint8)
        
        return {
            'anomaly_map': anomaly_map,
            'anomaly_mask': anomaly_mask,
            'reconstructed': outputs['reconstructed'].squeeze().cpu().numpy(),
            'rti_attention': outputs['rti_attention'].squeeze().cpu().numpy(),
            'hsi_attention': outputs['hsi_attention'].squeeze().cpu().numpy(),
            'fused_features': outputs['fused_features'].squeeze().cpu().numpy()
        }
    
    def visualize_results(self, results, save_folder=None):
        """
        Visualiser les résultats de détection d'anomalies
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Carte d'anomalie
        im1 = axes[0,0].imshow(results['anomaly_map'], cmap='hot')
        axes[0,0].set_title('Carte d\'anomalie')
        axes[0,0].axis('off')
        plt.colorbar(im1, ax=axes[0,0])
        
        # Masque binaire
        axes[0,1].imshow(results['anomaly_mask'], cmap='gray')
        axes[0,1].set_title('Masque d\'anomalie (binaire)')
        axes[0,1].axis('off')
        
        # Image reconstruite
        im3 = axes[0,2].imshow(results['reconstructed'], cmap='viridis')
        axes[0,2].set_title('Reconstruction')
        axes[0,2].axis('off')
        plt.colorbar(im3, ax=axes[0,2])
        
        # Attention RTI (moyenne des channels)
        rti_att_mean = np.mean(results['rti_attention'], axis=0)
        im4 = axes[1,0].imshow(rti_att_mean, cmap='Blues')
        axes[1,0].set_title('Attention RTI')
        axes[1,0].axis('off')
        plt.colorbar(im4, ax=axes[1,0])
        
        # Attention HSI (moyenne des channels)
        hsi_att_mean = np.mean(results['hsi_attention'], axis=0)
        im5 = axes[1,1].imshow(hsi_att_mean, cmap='Reds')
        axes[1,1].set_title('Attention HSI')
        axes[1,1].axis('off')
        plt.colorbar(im5, ax=axes[1,1])
        
        # Overlay anomalie sur reconstruction
        overlay = results['reconstructed'].copy()
        anomaly_colored = plt.cm.hot(results['anomaly_map'])[:,:,:3]
        alpha = results['anomaly_map']
        
        # Créer l'overlay
        for i in range(3):
            overlay_rgb = plt.cm.viridis(overlay)[:,:,:3] if i == 0 else overlay_rgb
        
        axes[1,2].imshow(overlay_rgb)
        axes[1,2].imshow(anomaly_colored, alpha=alpha*0.7)
        axes[1,2].set_title('Overlay Anomalies')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        
        if save_folder:
            plt.savefig(os.path.join(save_folder, 'anomaly_detection_results.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Statistiques
        print(f"📊 Statistiques de détection:")
        print(f"  • Pixels anomaliques détectés: {np.sum(results['anomaly_mask'])}")
        print(f"  • Pourcentage d'anomalies: {np.mean(results['anomaly_mask'])*100:.2f}%")
        print(f"  • Score d'anomalie moyen: {np.mean(results['anomaly_map']):.4f}")
        print(f"  • Score d'anomalie max: {np.max(results['anomaly_map']):.4f}")
    
    def save_model(self, save_path):
        """Sauvegarder le modèle entraîné"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'metadata': self.metadata if hasattr(self, 'metadata') else None
        }, save_path)
        print(f"✓ Modèle sauvé: {save_path}")
    
    def load_model(self, model_path, rti_dim=64, hsi_dim=64):
        """Charger un modèle pré-entraîné"""
        self.initialize_model(rti_dim, hsi_dim)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'metadata' in checkpoint:
            self.metadata = checkpoint['metadata']
        print(f"✓ Modèle chargé: {model_path}")

# ==================== EXEMPLE D'UTILISATION ====================
def run_anomaly_detection_pipeline(results_folder):
    """
    Pipeline complet de détection d'anomalies (optimisé CPU)
    """
    print("🚀 Démarrage du pipeline de détection d'anomalies")
    
    # Initialiser le pipeline en CPU
    pipeline = AnomalyDetectionPipeline(device='cuda')
    
    # Charger les features
    rti_features, hsi_features = pipeline.load_features(results_folder)
    
    # Initialiser le modèle
    pipeline.initialize_model(rti_features.shape[1], hsi_features.shape[1])
    
    # Entraîner le modèle (epochs réduits pour CPU)
    print("⏱️  Entraînement optimisé pour CPU (peut prendre quelques minutes)...")
    losses = pipeline.train_unsupervised(rti_features, hsi_features, epochs=100)
    
    # Détecter les anomalies
    print("🔍 Détection des anomalies...")
    results = pipeline.detect_anomalies(rti_features, hsi_features, threshold=0.4)
    
    # Créer dossier de sortie
    output_folder = os.path.join(results_folder, "anomaly_detection")
    os.makedirs(output_folder, exist_ok=True)
    
    # Visualiser les résultats
    print("📊 Génération des visualisations...")
    pipeline.visualize_results(results, output_folder)
    
    # Sauvegarder le modèle
    model_path = os.path.join(output_folder, "anomaly_detection_model.pth")
    pipeline.save_model(model_path)
    
    # Sauvegarder les résultats
    np.save(os.path.join(output_folder, "anomaly_map.npy"), results['anomaly_map'])
    np.save(os.path.join(output_folder, "anomaly_mask.npy"), results['anomaly_mask'])
    
    # Graphique des pertes d'entraînement
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Évolution de la perte pendant l\'entraînement (CPU)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'training_losses.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Estimation des performances
    print(f"\n⚡ Informations de performance:")
    print(f"  • Device utilisé: {pipeline.device}")

    print(f"✅ Pipeline terminé ! Résultats dans: {output_folder}")
    
    return pipeline, results

# ==================== UTILISATION OPTIMISÉE CPU ====================

#UTILISATION :

# Remplace par ton dossier de résultats
results_folder = "D:/project/meh/results_20250729_151145/"

# Lance le pipeline (optimisé CPU)
pipeline, results = run_anomaly_detection_pipeline(results_folder)
'''
# Si tu veux réduire encore plus le temps d'entraînement :
pipeline = AnomalyDetectionPipeline(device='cpu')
rti_features, hsi_features = pipeline.load_features(results_folder)
pipeline.initialize_model(rti_features.shape[1], hsi_features.shape[1])
losses = pipeline.train_unsupervised(rti_features, hsi_features, epochs=15)  # Encore plus rapide
results = pipeline.detect_anomalies(rti_features, hsi_features, threshold=0.3)

'''