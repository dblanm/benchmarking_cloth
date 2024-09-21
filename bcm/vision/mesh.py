import os

import numpy as np

import torch

from PIL import Image


class Mesh:
    def __init__(self, mesh_file, device, texture_file=None):
        mesh_file_ext = os.path.splitext(mesh_file)[1]
        if mesh_file_ext == ".obj":
            verts, faces = self._parse_obj(mesh_file)
        else:
            raise ValueError(
                f"File extension {mesh_file_ext} for file {mesh_file} not recognized."
            )
        self.texture_img = (
            np.array(Image.open(texture_file)) if texture_file is not None else None
        )

        self.n_verts = len(verts)
        # Convert from 1-based to 0-based
        self.faces = np.array(faces) - 1
        self.torch_vertices = torch.FloatTensor(verts).to(device)
        self.control_points = self._get_control_points()
        self.torch_control_points = (
            torch.from_numpy(self.control_points).float().to(device)
        )

    def _get_control_points(self):
        """
        Assume vertices are in order
        0   1 2 3 ... n-1
        n   n+1   ... 2n-1
          .   .   .   .
          .   .   .   .
          .   .   .   .
        n+m n+m+1 ... nm-1
        """
        # Build adjacency matrix of graph induced by faces
        # The (i, j) coordinate indicates if (i, j) is an edge (1) or not (0)
        # We consider edges as those pairs of vertices that are contiguous in the face (e.g. if face is [1, 2, 3]
        # edges are [1, 2], [2, 3], and [3, 1], where the order does not matter).
        adjacency_matrix = np.zeros((self.n_verts, self.n_verts))
        for t in self.faces:
            for i in range(len(t)):
                idx1, idx2 = t[i], t[(i + 1) % len(t)]
                adjacency_matrix[idx1, idx2] = 1
                adjacency_matrix[idx2, idx1] = 1

        assert (
            np.sum(adjacency_matrix[0]) == 2
        ), "First vertex does not correspond to a corner"
        corners = np.where(adjacency_matrix.sum(axis=0) == 2)[0]
        n = corners[1] + 1
        m = (corners[-1] + 1) / n
        assert n == int(n) and m == int(m), "Could not properly recover mesh dimensions"
        u, v = np.meshgrid(np.arange(n), np.arange(m))
        return np.c_[u.ravel(), v.ravel()] / [n - 1, m - 1]

    @staticmethod
    def _parse_obj(file_name):
        verts, faces = [], []

        with open(file_name, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        for line in lines:
            tokens = line.strip().split()
            if line.startswith("v "):  # Line is a vertex.
                vert = [float(x) for x in tokens[1:4]]
                if len(vert) != 3:
                    msg = "Vertex %s does not have 3 values. Line: %s"
                    raise ValueError(msg % (str(vert), str(line)))
                verts.append(vert)
            elif line.startswith("f "):  # Line is a face.
                # Update face properties info.
                face = tokens[1:]
                face_list = [f.split("/") for f in face]
                face_verts = []

                for vert_props in face_list:
                    # Vertex index.
                    face_verts.append(int(vert_props[0]))
                    if len(vert_props) > 3:
                        raise ValueError(
                            f"Face vertices can only have 3 properties. \
                                        Face vert {vert_props}, Line: {line}"
                        )
                faces.append(face_verts)
        return verts, faces

    def __len__(self):
        return self.n_verts
