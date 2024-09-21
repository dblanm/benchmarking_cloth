from pathlib import Path

import numpy as np
import pybullet

ANCHOR_MIN_DIST = 0.02  # 2cm
ANCHOR_MASS = 0.100  # 100g
ANCHOR_RADIUS = 0.05  # 5cm
ANCHOR_RGBA_ACTIVE = (1, 0, 1, 1)  # magenta
ANCHOR_RGBA_INACTIVE = (0.5, 0.5, 0.5, 1)  # gray


def load_deform_object(
    sim,
    obj_file_name,
    texture_file_name,
    scale,
    mass,
    init_pos,
    init_ori,
    bending_stiffness,
    damping_stiffness,
    elastic_stiffness,
    friction_coeff,
    self_collision,
    debug,
):
    """Load object from obj file with pybullet's loadSoftBody()."""
    if debug:
        print("Loading filename", obj_file_name)
    # Note: do not set very small mass (e.g. 0.01 causes instabilities).
    deform_id = sim.loadSoftBody(
        fileName=str(Path(obj_file_name)),
        basePosition=init_pos,
        baseOrientation=pybullet.getQuaternionFromEuler(init_ori),
        scale=scale,
        mass=mass,  # 1kg is default; bad sim with lower mass
        collisionMargin=0.003,  # how far apart do two objects begin interacting
        useSelfCollision=self_collision,
        useBendingSprings=True,
        useFaceContact=True,
        useMassSpring=True,
        # springDampingAllDirections=True,
        springDampingAllDirections=1,
        springElasticStiffness=elastic_stiffness,
        springDampingStiffness=damping_stiffness,
        springBendingStiffness=bending_stiffness,
        frictionCoeff=friction_coeff,
        # useNeoHookean=0,
        useNeoHookean=False,
        # repulsionStiffness=10000000,
    )
    # PyBullet examples for loading and anchoring deformables:
    # github.com/bulletphysics/bullet3/examples/pybullet/examples/deformable_anchor.py
    sim.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
    kwargs = {}
    if hasattr(pybullet, "VISUAL_SHAPE_DOUBLE_SIDED"):
        kwargs["flags"] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED
    if texture_file_name is not None:
        kwargs["textureUniqueId"] = sim.loadTexture(str(Path(texture_file_name)))
    sim.changeVisualShape(deform_id, -1, rgbaColor=[1, 1, 1, 1], **kwargs)
    num_mesh_vertices = get_mesh_data(sim, deform_id)[0]

    if debug:
        print(
            "Loaded deform_id",
            deform_id,
            "with",
            num_mesh_vertices,
            "mesh vertices",
            "init_pos",
            init_pos,
        )
    # Pybullet will struggle with very large meshes, so we should keep mesh
    # sizes to a limited number of vertices and faces.
    # Large meshes will load on Linux/Ubuntu, but sim will run too slowly.
    # Meshes with >2^13=8196 vertices will fail to load on OS X due to shared
    # memory limits, as noted here:
    # github.com/bulletphysics/bullet3/issues/1965
    assert num_mesh_vertices < 2**13  # make sure mesh has less than ~8K verts
    return deform_id


def get_mesh_data(sim, deform_id):
    """Returns num mesh vertices and vertex positions."""
    kwargs = {}
    if hasattr(pybullet, "MESH_DATA_SIMULATION_MESH"):
        kwargs["flags"] = pybullet.MESH_DATA_SIMULATION_MESH
    num_verts, mesh_vert_positions = sim.getMeshData(deform_id, **kwargs)
    return num_verts, mesh_vert_positions


def attach_anchor(sim, anchor_id, anchor_vertices, deform_id, change_color=False):
    if change_color:
        sim.changeVisualShape(anchor_id, -1, rgbaColor=ANCHOR_RGBA_ACTIVE)
    for v in anchor_vertices:
        sim.createSoftBodyAnchor(deform_id, v, anchor_id, -1)


def create_anchor(
    sim,
    anchor_pos,
    anchor_idx,
    preset_vertices,
    mesh,
    radius,
    mass=0.1,
    rgba=(1, 0, 1, 1.0),
    use_preset=True,
    use_closest=True,
):
    """
    Create an anchor in Pybullet to grab or pin an object.
    :param sim: The simulator object
    :param anchor_pos: initial anchor position
    :param anchor_idx: index of the anchor (0:left, 1:right ...)
    :param preset_vertices: a preset list of vertices for the anchors
                            to grab on to (if use_preset is enabled)
    :param mesh: mesh of the deform object
    :param mass: mass of the anchor
    :param radius: visual radius of the anchor object
    :param rgba: color of the anchor
    :param use_preset: Use preset of anchor vertices
    :param use_closest: Use closest vertices to anchor as grabbing vertices
           (if no preset is used), ensuring anchors
    has something to grab on to
    :return: Anchor's ID, anchor's position, anchor's vertices
    """
    anchor_vertices = None
    mesh = np.array(mesh)
    if use_preset and preset_vertices is not None:
        anchor_vertices = preset_vertices[anchor_idx]
        anchor_pos = mesh[anchor_vertices].mean(axis=0)
    elif use_closest:
        anchor_pos, anchor_vertices = get_closest(anchor_pos, mesh)
    anchor_geom_id = create_anchor_geom(sim, anchor_pos, mass, radius, rgba)
    return anchor_geom_id, anchor_pos, anchor_vertices


def create_anchor_geom(sim, pos, mass, radius, rgba, use_collision=True):
    """Create a small visual object at the provided pos in world coordinates.
    If mass==0: the anchor will be fixed (not moving)
    If use_collision==False: this object does not collide with any other objects
    and would only serve to show grip location.
    input: sim (pybullet sim), pos (list of 3 coords for anchor in world frame)
    output: anchorId (long) --> unique bullet ID to refer to the anchor object
    """
    anchor_visual_shape = sim.createVisualShape(
        pybullet.GEOM_SPHERE, radius=radius, rgbaColor=rgba
    )
    if mass > 0 and use_collision:
        anchor_collision_shape = sim.createCollisionShape(
            pybullet.GEOM_SPHERE, radius=radius
        )
    else:
        anchor_collision_shape = -1
    anchor_id = sim.createMultiBody(
        baseMass=mass,
        basePosition=pos,
        baseCollisionShapeIndex=anchor_collision_shape,
        baseVisualShapeIndex=anchor_visual_shape,
        useMaximalCoordinates=True,
    )
    return anchor_id


def get_closest(init_pos, mesh, max_dist=None):
    """Find mesh points closest to the given point."""
    init_pos = np.array(init_pos).reshape(1, -1)
    mesh = np.array(mesh)
    # num_pins_per_pt = max(1, mesh.shape[0] // 50)
    # num_to_pin = min(mesh.shape[0], num_pins_per_pt)
    num_to_pin = 1  # new pybullet behaves well with 1 vertex per anchor
    dists = np.linalg.norm(mesh - init_pos, axis=1)
    anchor_vertices = np.argpartition(dists, num_to_pin)[0:num_to_pin]
    if max_dist is not None:
        anchor_vertices = anchor_vertices[dists[anchor_vertices] <= max_dist]
    new_anc_pos = mesh[anchor_vertices].mean(axis=0)
    return new_anc_pos, anchor_vertices
