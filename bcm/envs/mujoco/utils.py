import xml.etree.ElementTree as ET

from ..utils import get_faces_regular_mesh


class XMLModel:
    def __init__(self, xml_file, is_v3):
        self.path = xml_file
        self.tree = ET.parse(self.path)
        self.mujoco_is_v3 = is_v3
        if self.mujoco_is_v3:
            self.cloth = next(self.tree.iter("flexcomp"))
        else:
            self.cloth = next(self.tree.iter("composite"))

    def save_changes_to_file(self):
        with open(self.path, "wb") as f:
            self.tree.write(f, encoding="utf-8")

    def modify_params(self, params):
        # Keys of params either of the form
        # "key" or "key_subkey" (underscore is important)
        # e.g., "damping" or "joint_damping"
        if self.mujoco_is_v3:
            self.modify_params_v_3_or_more(params)
        else:
            self.modify_params_v_less_than_3(params)

    def modify_params_v_3_or_more(self, params):
        for k, val in params.items():
            if "_" in k:
                *subelements, subkey = k.split("_")
                root = [self.cloth]
                for subelement in subelements:
                    if len(root) > 1:
                        for r in root:
                            if r.get("key") == subelement:
                                root = [r]
                                break
                    elif len(root) == 1:
                        root = root[0].findall(subelement)
                    else:
                        raise ValueError(f"Cannot modify {k}: Got root {root}")
                assert (
                    len(root) == 1
                ), f"Found non-unique element for {k}: Got root {root}"
                root[0].set(subkey, str(val))
            else:
                self.cloth.set(k, str(val))
        self.save_changes_to_file()

    def modify_params_v_less_than_3(self, params):
        for k, val in params.items():
            if "_" in k:
                subelement, subkey = k.split("_")
                self.cloth.find(subelement).set(subkey, str(val))
            else:
                self.cloth.set(k, str(val))
        self.save_changes_to_file()

    def change_texture(self, texture_file):
        for text in self.tree.iter("texture"):
            if text.attrib["name"] == "cloth_texture":
                text.set("file", texture_file)
                return
        raise ValueError("Could not change texture")

    def get_cloth_size(self):
        num_rows, num_cols, _ = map(int, self.cloth.attrib["count"].split())
        return num_rows, num_cols

    def get_mesh_ids(self, model):
        if self.mujoco_is_v3:
            faces = model.flex_elem.reshape(-1, model.flex_dim[0] + 1)
            return model.flex_vertbodyid, faces
        else:
            n_rows, n_cols = self.get_cloth_size()
            verts_ids = []
            faces = get_faces_regular_mesh(n_rows=n_rows, n_cols=n_cols)
            for row in range(n_rows):
                for col in range(n_cols):
                    verts_ids.append(model.geom(f"CG{row}_{col}").bodyid[0])
            return verts_ids, faces
