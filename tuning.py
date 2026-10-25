"""Ajustes de MLX medidos en el M4. Importar ANTES que mlx (el valor se lee al crear el dispositivo Metal).

MLX_MAX_OPS_PER_BUFFER=10: cerrar los command buffers cada 10 operaciones deja que la GPU empiece antes.
Con el valor por defecto (se comporta como ~20) una pasada de 1 token tarda 56 ms y una de 3 tokens 70;
con 10, 52 y 62 ms. En el banco intercalado: MTP k=2 27,5 -> 28,8 tok/s. 2 y 5 son algo peores que 10.
"""
import os

os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "10")
