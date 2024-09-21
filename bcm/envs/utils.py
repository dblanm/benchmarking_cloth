import os


def get_faces_regular_mesh(n_rows, n_cols):
    faces = []
    for row in range(n_rows - 1):
        for col in range(n_cols - 1):
            idx = row * n_cols + col
            faces.append([idx, idx + 1, idx + 1 + n_cols])
            faces.append([idx, idx + 1 + n_cols, idx + n_cols])
    return faces


def get_texture_file_name(target, assets_path):
    file_name = os.path.join(
        assets_path, "textures", target.rpartition("_")[0] + ".png"
    )
    assert os.path.isfile(file_name), f"File {file_name} does not exist"
    return file_name
