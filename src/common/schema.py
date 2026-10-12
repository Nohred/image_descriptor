"""Contrato compartido entre detección, descriptores y base de datos.

Un registro por objeto detectado:
  object_id, image_id, image_path, bbox_xyxy,
  attributes {color, vehicle_type}, scores,
  raw_description, model_versions, experiment_id
"""
