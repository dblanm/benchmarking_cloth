from .cloth_dyn_sofa import ClothDynSofa


def get_cloth_sofa_env(**kwargs):
    return ClothDynSofa(**kwargs)
