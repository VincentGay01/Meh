import cv2
import numpy as np
import os
from pathlib import Path

def calculate_homography(img_reference, img_to_align):
    """
    Calcule l'homographie entre l'image de référence et l'image à aligner.
    
    Args:
        img_reference: Image de référence (basse résolution)
        img_to_align: Image à aligner (haute résolution)
    
    Returns:
        H: Matrice d'homographie
        target_size: Taille cible (width, height)
    """
    # Redimensionner l'image haute résolution à la taille de la référence
    target_size = (img_reference.shape[1], img_reference.shape[0])
    img_resized = cv2.resize(img_to_align, target_size, interpolation=cv2.INTER_AREA)
    
    # Convertir en niveaux de gris
    gray_ref = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)
    gray_align = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # AKAZE pour détection + descripteurs
    akaze = cv2.AKAZE_create(threshold=1e-4)
    kp1, des1 = akaze.detectAndCompute(gray_ref, None)
    kp2, des2 = akaze.detectAndCompute(gray_align, None)
    
    if des1 is None or des2 is None:
        raise RuntimeError("Impossible de détecter des points caractéristiques.")
    
    # Matcher avec Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Ratio test (Lowe)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.85 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 10:
        raise RuntimeError(f"Pas assez de correspondances pour estimer l'homographie. Trouvé: {len(good_matches)}")
    
    # Points pour homographie
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Calculer l'homographie
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    
    return H, target_size

def apply_homography(img, H, target_size):
    """
    Applique une homographie à une image.
    
    Args:
        img: Image à transformer
        H: Matrice d'homographie
        target_size: Taille de sortie (width, height)
    
    Returns:
        Image transformée
    """
    # Redimensionner d'abord l'image à la taille cible
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Appliquer la transformation
    aligned_img = cv2.warpPerspective(img_resized, H, target_size)
    
    return aligned_img

def align_multiple_images(reference_path, images_to_align_paths, output_dir="aligned_output"):
    """
    Aligne plusieurs images sur une image de référence.
    
    Args:
        reference_path: Chemin vers l'image de référence
        images_to_align_paths: Liste des chemins vers les images à aligner
        output_dir: Dossier de sortie
    """
    # Créer le dossier de sortie
    Path(output_dir).mkdir(exist_ok=True)
    
    # Charger l'image de référence
    img_reference = cv2.imread(reference_path)
    if img_reference is None:
        raise FileNotFoundError(f"Impossible de charger l'image de référence: {reference_path}")
    
    # Sauvegarder l'image de référence dans le dossier de sortie
    cv2.imwrite(os.path.join(output_dir, "reference.jpg"), img_reference)
    
    print(f"Image de référence chargée: {reference_path}")
    print(f"Nombre d'images à aligner: {len(images_to_align_paths)}")
    
    # Calculer l'homographie avec la première image
    first_img = cv2.imread(images_to_align_paths[0])
    if first_img is None:
        raise FileNotFoundError(f"Impossible de charger la première image: {images_to_align_paths[0]}")
    
    print("Calcul de l'homographie...")
    H, target_size = calculate_homography(img_reference, first_img)
    
    # Appliquer l'homographie à toutes les images
    aligned_images = []
    for i, img_path in enumerate(images_to_align_paths):
        print(f"Alignement de l'image {i+1}/{len(images_to_align_paths)}: {img_path}")
        
        # Charger l'image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Attention: Impossible de charger {img_path}, ignoré.")
            continue
        
        # Appliquer l'homographie
        try:
            aligned_img = apply_homography(img, H, target_size)
            aligned_images.append(aligned_img)
            
            # Sauvegarder l'image alignée
            output_filename = f"rti_{i}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, aligned_img)
                        
        except Exception as e:
            print(f"Erreur lors de l'alignement de {img_path}: {e}")
            continue
    
    print(f"Alignement terminé! {len(aligned_images)} images traitées.")
    print(f"Résultats sauvegardés dans: {output_dir}")
    
    return aligned_images

def main():
    """
    Fonction principale - modifiez les chemins selon vos besoins
    """
    # === Configuration ===
    reference_image = 'D:/project/meh/404.png'  # Image de référence (basse résolution)
    """
    # Liste des images à aligner (modifiez selon vos besoins)
    images_to_align = [
        'D:/project/meh/DSC_0001.jpg',
        'D:/project/meh/DSC_0002.jpg',
        'D:/project/meh/DSC_0003.jpg',
        # Ajoutez d'autres images ici
    ]
    """
    # Ou charger toutes les images d'un dossier
    input_folder = 'D:/project/meh/images_to_align'
    images_to_align = [str(p) for p in Path(input_folder).glob('*.jpg')]
    images_to_align.extend([str(p) for p in Path(input_folder).glob('*.png')])
    
    output_directory = 'D:/project/meh/aligned_results'
    
    try:
        # Aligner toutes les images
        aligned_images = align_multiple_images(reference_image, images_to_align, output_directory)
        
        # Affichage optionnel des résultats
        if aligned_images:
            print("\nAffichage des résultats (appuyez sur une touche pour passer à la suivante)...")
            
            # Afficher l'image de référence
            img_ref = cv2.imread(reference_image)
            cv2.imshow("Image de référence", img_ref)
            cv2.waitKey(0)
            
            # Afficher chaque image alignée
            for i, aligned_img in enumerate(aligned_images):
                cv2.imshow(f"Image alignée {i+1}", aligned_img)
                
                # Superposition
                overlay = cv2.addWeighted(img_ref, 0.5, aligned_img, 0.5, 0)
                cv2.imshow(f"Superposition {i+1}", overlay)
                
                cv2.waitKey(0)
            
            cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()