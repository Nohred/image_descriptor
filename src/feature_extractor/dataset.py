import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

class VeRiDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Ruta al archivo CSV con las anotaciones (imageName, colorID, typeID).
            img_dir (str): Directorio con las imágenes originales recortadas.
            transform (callable, optional): Transformaciones opcionales a aplicar a la imagen.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Obtener el nombre de la imagen y construir la ruta
        img_name = self.data_frame.iloc[idx]['imageName']
        img_path = os.path.join(self.img_dir, img_name)

        # 2. Cargar la imagen (convertida a RGB por seguridad)
        image = Image.open(img_path).convert('RGB')

        # 3. Obtener las etiquetas (ya están en base 0)
        color_id = int(self.data_frame.iloc[idx]['colorID'])
        type_id = int(self.data_frame.iloc[idx]['typeID'])

        # 4. Aplicar transformaciones (Resize, tensores, normalización)
        if self.transform:
            image = self.transform(image)

        # PyTorch espera que las etiquetas de clasificación sean tensores de tipo 'long' (enteros de 64 bits)
        return image, torch.tensor(color_id, dtype=torch.long), torch.tensor(type_id, dtype=torch.long)


def get_dataloaders(
    train_csv,
    train_dir,
    test_csv,
    test_dir,
    batch_size=32,
    num_workers=4,
    val_split=0.2,
    seed=42,
):
    """
    Construye los DataLoaders de entrenamiento, validación y prueba.

    La validación se separa únicamente del CSV de entrenamiento. El CSV de
    prueba se mantiene intacto para la evaluación final.
    """
    
    # Transformaciones para ENTRENAMIENTO (Incluye Data Augmentation)
    train_transform = T.Compose([
        T.Resize((224, 224)),               # Tamaño requerido por EfficientNet
        T.RandomHorizontalFlip(p=0.5),      # Volteo horizontal al azar (ayuda a generalizar)
        T.ColorJitter(brightness=0.2, contrast=0.2), # Ligeros cambios de luz
        T.ToTensor(),                       # Convierte de PIL a Tensor PyTorch (0 a 1)
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalización ImageNet
    ])

    # Transformaciones para PRUEBA (Sin Augmentation, solo formato estricto)
    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not 0 < val_split < 1:
        raise ValueError("val_split debe estar entre 0 y 1.")

    # Se crean datasets separados para que validación no use augmentation.
    train_dataset = VeRiDataset(csv_file=train_csv, img_dir=train_dir, transform=train_transform)
    val_dataset = VeRiDataset(csv_file=train_csv, img_dir=train_dir, transform=test_transform)
    test_dataset = VeRiDataset(csv_file=test_csv, img_dir=test_dir, transform=test_transform)

    if len(train_dataset) < 2:
        raise ValueError("Se necesitan al menos 2 imágenes en el CSV de entrenamiento.")

    val_size = max(1, int(round(len(train_dataset) * val_split)))
    train_size = len(train_dataset) - val_size
    if train_size < 1:
        raise ValueError("El split deja menos de una muestra para entrenamiento.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Ambos subconjuntos provienen solo del CSV de entrenamiento.
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

# --- BLOQUE DE PRUEBA RÁPIDA ---
if __name__ == "__main__":
    # Prueba de sanidad para asegurarnos de que todo carga correctamente
    train_csv = "data/train_features.csv"
    train_dir = "../VeRi/image_train"
    
    try:
        dataset = VeRiDataset(train_csv, train_dir)
        imagen, color, tipo = dataset[0]
        print(f"[+] Dataset cargado exitosamente.")
        print(f"    Total de imágenes: {len(dataset)}")
        print(f"    Muestra 0 - Tamaño original: {imagen.size}, Color: {color}, Tipo: {tipo}")
    except Exception as e:
        print(f"[-] Error cargando el dataset: {e}")