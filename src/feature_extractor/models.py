import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskVehicleNet(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', num_colors=10, num_types=9, pretrained=True):
        """
        Red Multi-Tarea para extraer características y clasificar Color y Tipo.
        
        Args:
            backbone_name (str): 'efficientnet_b0' o 'mobilenet_v3_small'.
            num_colors (int): Número de clases de color.
            num_types (int): Número de clases de tipo de vehículo.
            pretrained (bool): Usar pesos pre-entrenados de ImageNet (Recomendado).
        """
        super(MultiTaskVehicleNet, self).__init__()
        self.backbone_name = backbone_name
        
        # ---------------------------------------------------------
        # 1. Cargar el Backbone pre-entrenado
        # ---------------------------------------------------------
        if backbone_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            
            # Obtener el tamaño de salida (1280 para EfficientNet-B0)
            in_features = self.backbone.classifier[1].in_features
            
            # Reemplazar el clasificador original con una capa de Identidad
            # Esto hace que el backbone devuelva directamente el vector de características de 1280
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'mobilenet_v3_small':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            
            # Obtener el tamaño de salida (576 para MobileNetV3-Small)
            in_features = self.backbone.classifier[0].in_features
            
            # El backbone devuelve el vector base antes de la proyección del clasificador.
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError("Backbone no soportado. Usa 'efficientnet_b0' o 'mobilenet_v3_small'")

        # ---------------------------------------------------------
        # 2. Definir las Cabezas de Clasificación (Cabezas Multi-Tarea)
        # ---------------------------------------------------------
        # Cabeza para Color
        self.color_head = nn.Sequential(
            nn.Dropout(p=0.2), # Previene sobreajuste
            nn.Linear(in_features, num_colors)
        )
        
        # Cabeza para Tipo
        self.type_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_types)
        )

    def forward(self, x):
        """
        Paso hacia adelante de la red.
        Recibe una imagen (tensor) y devuelve las predicciones de color y tipo,
        ADEMÁS del vector de características puras.
        """
        # 1. Extraer características base
        features = self.backbone(x)
        
        # 2. Pasar por las dos cabezas independientes
        color_preds = self.color_head(features)
        type_preds = self.type_head(features)
        
        # Devolvemos 3 cosas: 
        # - Predicciones de color (para loss/métricas)
        # - Predicciones de tipo (para loss/métricas)
        # - Features (el vector puro, ideal si luego quieres usar esto como descriptor)
        return color_preds, type_preds, features


# --- BLOQUE DE PRUEBA RÁPIDA ---
if __name__ == "__main__":
    # Crear un tensor de prueba que simule 1 imagen de 224x224 (Batch, Canales, Alto, Ancho)
    # PyTorch espera tensores [B, C, H, W]
    tensor_prueba = torch.randn(1, 3, 224, 224)

    # 1. Probar EfficientNet
    print("--- Probando EfficientNet-B0 ---")
    modelo_eff = MultiTaskVehicleNet(backbone_name='efficientnet_b0')
    out_color, out_type, out_features = modelo_eff(tensor_prueba)
    print(f"Salida Color: {out_color.shape} (Esperado: 1, 10)")
    print(f"Salida Tipo : {out_type.shape} (Esperado: 1, 9)")
    print(f"Features puras: {out_features.shape} (Esperado: 1, 1280)")

    print("\n--- Probando MobileNetV3-Small ---")
    modelo_mob = MultiTaskVehicleNet(backbone_name='mobilenet_v3_small')
    out_color, out_type, out_features = modelo_mob(tensor_prueba)
    print(f"Salida Color: {out_color.shape} (Esperado: 1, 10)")
    print(f"Salida Tipo : {out_type.shape} (Esperado: 1, 9)")
    print(f"Features puras: {out_features.shape} (Esperado: 1, 576)")