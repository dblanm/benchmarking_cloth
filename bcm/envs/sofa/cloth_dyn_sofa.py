import faulthandler
import os

import numpy as np
import Sofa
import SofaRuntime


class ClothDynSofa:
    def __init__(
        self, root=None, dt=1 / 100, max_steps=300, params={}, real_setup={}, **kwargs
    ):
        self.params = params
        faulthandler.enable()
        self.add_edge = False
        self.dt = dt
        self.trajectory_dt = self.dt / 10
        # max_steps, how long the simulator should run. Total length: dt*max_steps
        self.max_steps = max_steps
        self.scale = params["scale"]
        self.real_setup = real_setup

        self.ode_solver_threshold = 1e-100
        self.ode_solver_tolerance = 1e-09
        self.ode_solver_iter = 25

        self.position_grippers = np.array([0.0, 0.0, 1.0, 0.5, 0.0, 1.0]) * self.scale
 
        # root node in the simulator
        if root is None:
            self.root = Sofa.Core.Node("myroot")
        else:
            self.root = root
        self.mesh_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "../../assets/meshes"
            )
        )
        print(f"Mesh path={self.mesh_path}")

        self.root.findData("dt").value = dt
        self.table_z_pos = 1.02  # 0.8 = 0.08? - Change from 1.0 as there is a small mismatch

        # the current step in the simulation
        self.current_step = 0

        self.load_plugins()

        self.place_objects_in_scene()

        self.actuators = [self.Input.getObject("DOFs"), self.Input_2.getObject("DOFs2")]

        self.ballMesh = [self.ball, self.ball_2]
        self.min_dist = 5.0  # correction for the ball motion anomaly
        self.numVertices = 4
        self.positive_rewards = False
        self.gripped = True
        self.distance_threshold = 30.0
        self.workspace = np.array([0, 0, -60])  # restrictions on x, y, z
        self.restricted_workspace = True
        self.give_relative_pos = (
            False  # provide relative positions of the vertices in the state space
        )
        self.domain_randomization = 1
        self.dynamic_randomization = 0
        self.edge_observations = True
        self.explicit_policy = False
        self.edge_large = False

        # self.place_camera()

        # if creating_root:
        #     self.start_simulation()

        self.tmp_file = "/tmp/img.png"

    def place_camera(self):
        cam_lookAt = np.array([0.0176, 0.79, 0.78]) * self.scale
        distance = 4 * self.scale
        # orientation = [0.046, 0.987, 0.048, 0.14366]
        position = [-60.3223, 80.742, 288.598]
        orientation = [-0.0264115, -0.20198, -0.0404923, 0.979743]
        fov = 45
        zNear = 6.21
        zFar = 480
        self.root.addObject(
            "InteractiveCamera",
            name="camera",
            orientation=orientation,
            lookAt=cam_lookAt.tolist(),
            distance=distance,
            position=position,
            # fieldOfView=66,
            # zNear=-10,
            # zFar=300,
            fieldOfView=fov,
            zNear=zNear,
            zFar=zFar,
        )

    def _setup_gripper_trajectory(
        self,
        trajectory,
        pre_traj_length,
        compute_metrics_fn,
        targets,
        target_it,
        target_name,
        select_targets,
    ):
        self.root.addObject(
            ClothPickers(
                cloth=self.SquareGravity,
                actuators=self.actuators,
                name="ClotPickers",
                scale=self.scale,
                dt=self.dt,
                trajectory=trajectory,
                pre_traj_length=pre_traj_length,
                compute_metrics_fn=compute_metrics_fn,
                targets=targets,
                target_it=target_it,
                target_name=target_name,
                select_targets=select_targets,
                **self.real_setup,
            )
        )

    def start_simulation(
        self,
        trajectory,
        pre_traj_length,
        compute_metrics_fn,
        targets,
        target_it,
        target_name,
        select_targets,
    ):
        enable_gui = False
        # enable_gui = True

        # Define the cloth picker trajectory
        if enable_gui:
            self._setup_gripper_trajectory(
                trajectory,
                pre_traj_length,
                compute_metrics_fn,
                targets,
                target_it,
                target_name,
                select_targets,
            )

        # start the simulator
        Sofa.Simulation.init(self.root)
        # self.place_objects_in_scene()
        # start the gui
        # Sofa.Gui.GUIManager.Init("myscene", "qt")
        if enable_gui:
            Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
            Sofa.Gui.GUIManager.createGUI(self.root, __file__)
            Sofa.Gui.GUIManager.SetDimension(1080, 1080)
            Sofa.Gui.GUIManager.MainLoop(
                self.root
            )  # This will make to run everything from the GUI
        # Sofa.Gui.GUIManager.closeGUI()

    def close(self):
        Sofa.Simulation.unload(self.root)

    def load_plugins(self):
        plugins = [
            "Sofa.GL.Component.Rendering3D",
            "Sofa.GL.Component.Shader",
            "Sofa.Component.AnimationLoop",
            "Sofa.Component.Collision.Detection.Algorithm",
            "Sofa.Component.Collision.Detection.Intersection",
            "Sofa.Component.Collision.Geometry",
            "Sofa.Component.Constraint.Lagrangian.Correction",
            "Sofa.Component.Haptics",
            "Sofa.Component.IO.Mesh",
            "Sofa.Component.LinearSolver.Iterative",
            "Sofa.Component.Mass",
            "Sofa.Component.ODESolver.Backward",
            "Sofa.Component.SolidMechanics.FEM.Elastic",
            "Sofa.Component.SolidMechanics.Spring",
            "Sofa.Component.Constraint.Projective",
        ]
        for plugin in plugins:
            SofaRuntime.importPlugin(plugin)
            self.root.addObject("RequiredPlugin", name=plugin)

    @staticmethod
    def array2str(arr):
        return str(arr)[1:-1]

    def _place_cloth(self, pos_cloth):
        SquareGravity = self.root.addChild("SquareGravity")
        self.SquareGravity = SquareGravity
        SquareGravity.addObject(
            "EulerImplicitSolver", name="cg_odesolver", printLog="false"
        )
        SquareGravity.addObject(
            "CGLinearSolver",
            threshold=self.ode_solver_threshold,
            tolerance=self.ode_solver_tolerance,
            name="linear solver",
            iterations=self.ode_solver_iter,
        )
        # Add the cloth object
        # Pose should be pos="0.0 0.0 1.0"
        SquareGravity.addObject(
            "MeshObjLoader",
            scale=str(self.scale - 5),
            name="loader",
            translation=pos_cloth,
            createSubelements="true",
            filename=os.path.join(self.mesh_path, "old_rag.obj"),
        )
        SquareGravity.addObject(
            "TriangleSetTopologyContainer", src="@loader", name="topo"
        )
        SquareGravity.addObject(
            "TriangleSetGeometryAlgorithms",
            recomputeTrianglesOrientation="1",
            name="algo",
            template="Vec3d",
        )
        SquareGravity.addObject(
            "MechanicalObject", src="@loader", name="cloth", template="Vec3d"
        )
        # We can specify either the total mass or the vertex mass
        SquareGravity.addObject(
            "UniformMass", template="Vec3d", vertexMass=self.params.vertex_mass
        )
        # SquareGravity.addObject('DiagonalMass', name="UniformMass",
        #                         massDensity=self.params.vertex_mass)
        # SquareGravity.addObject('DiagonalMass', name="UniformMass",
        #                         totalMass=self.params.vertex_mass)
        SquareGravity.addObject(
            "TriangularFEMForceField",
            name="TriangularFEMForceField",
            youngModulus=self.params.youngmodulus,
            template="Vec3d",
            method="large",
            poissonRatio=self.params.poissonratio,
        )
        # For Cloth Dynamic mesh
        SquareGravity.addObject(
            "TriangularBendingSprings",
            template="Vec3d",
            damping=self.params.damping,
            stiffness=self.params.stiffness,
            name="FEM-Bend",
        )
        # For Cloth Static Mesh
        # SquareGravity.addObject(
        #     "MeshSpringForceField",
        #     template="Vec3d",
        #     # method="large",
        #     damping=self.params.damping,
        #     stiffness=self.params.stiffness,
        #     name="Springs",
        # )
        SquareGravity.addObject("UncoupledConstraintCorrection")

        # self.root/SquareGravity/TriangularSurface
        TriangularSurface = SquareGravity.addChild("TriangularSurface")
        self.TriangularSurface = TriangularSurface
        TriangularSurface.addObject("TriangleSetTopologyContainer", name="Container")
        TriangularSurface.addObject("TriangleSetTopologyModifier", name="Modifier")
        TriangularSurface.addObject("TriangleCollisionModel")
        TriangularSurface.addObject("LineCollisionModel")
        TriangularSurface.addObject("PointCollisionModel")

        # self.root/SquareGravity/clothVisual
        clothVisual = SquareGravity.addChild("clothVisual")
        self.clothVisual = clothVisual
        clothVisual.addObject(
            "OglModel",
            # color="red",
            src="@../loader",
            name="Visual",
            texturename=os.path.join(
                self.mesh_path, os.pardir, "textures", "rag_cloth.png"
            ),
        )
        clothVisual.addObject("IdentityMapping", input="@..", output="@Visual")

        if self.add_edge:
            self.place_edge()

    def _place_gripper_left(self, pos1, selected_corners):
        orientation = "0 0 0 1.0"
        pos_gripper_left = pos1 + " " + orientation
        Input = self.root.addChild("Input")
        # Add the First gripper
        self.Input = Input
        Input.addObject(
            "MechanicalObject",
            position=pos_gripper_left,
            name="DOFs",
            template="Rigid3d",
        )

        # self.root/Input/VisuAvatar
        VisuAvatar = Input.addChild("VisuAvatar")
        self.VisuAvatar = VisuAvatar
        VisuAvatar.activated = False
        VisuAvatar.addObject(
            "OglModel",
            color="blue",
            name="Visual",
            fileMesh=os.path.join(self.mesh_path, "sphere.obj"),
        )
        VisuAvatar.addObject("RigidMapping", input="@..", output="@Visual")

        # self.root/Input/RefModel
        RefModel = Input.addChild("RefModel")
        self.RefModel = RefModel
        RefModel.addObject(
            "MeshObjLoader",
            name="loader",
            filename=os.path.join(self.mesh_path, "sphere.obj"),
        )
        RefModel.addObject("Mesh", src="@loader")
        RefModel.addObject(
            "MechanicalObject", src="@loader", name="instrumentCollisionState"
        )
        RefModel.addObject("RigidMapping")

        # Create the object that is attached to the gripper
        # self.root/ball
        ball = self.root.addChild("ball")
        self.ball = ball
        ball.addObject(
            "EulerImplicitSolver",
            rayleighStiffness="0.05",
            name="ODE solver",
            rayleighMass="0.0",
        )
        ball.addObject(
            "CGLinearSolver",
            threshold="10e-20",
            tolerance="1e-10",
            name="linear solver",
            iterations="10",
        )
        ball.addObject(
            "MechanicalObject", translation=pos1, name="ballState", template="Rigid3d"
        )
        ball.addObject("UniformMass", name="mass", totalMass="0.00001")
        ball.addObject("LCPForceFeedback", activate="true", forceCoef="0.001")
        ball.addObject("UncoupledConstraintCorrection", compliance="100")

        # self.root/ball/ball_surf
        ball_surf = ball.addChild("ball_surf")
        self.ball_surf = ball_surf
        ball_surf.addObject(
            "MeshObjLoader",
            name="meshLoader",
            filename=os.path.join(self.mesh_path, "sphere.obj"),
            # scale=self.scale / 1000
        )
        ball_surf.addObject("Mesh", src="@meshLoader")
        ball_surf.addObject(
            "MechanicalObject",
            src="@meshLoader",
            name="ballCollisionState",
            template="Vec3d",
        )
        ball_surf.addObject("UniformMass", totalMass="0.00001")
        ball_surf.addObject("UncoupledConstraintCorrection", compliance="1")
        ball_surf.addObject("RigidMapping", input="@..", output="@ballCollisionState")

        # self.root/ball/ball_surf/VisualModel
        VisualModel = ball_surf.addChild("VisualModel")
        self.VisualModel = VisualModel
        VisualModel.addObject(
            "OglModel",
            color="1.0 0.2 0.2 1.0",
            src="@../meshLoader",
            name="ballVisualModel",
        )
        VisualModel.addObject("IdentityMapping", input="@..", output="@ballVisualModel")
        ball.addObject(
            "VectorSpringForceField",
            object1="@Input/RefModel/instrumentCollisionState",
            object2="@ball/ball_surf/ballCollisionState",
            viscosity="0",
            stiffness="100000",
        )
        # Attach the ball to the index 420 of the cloth
        self.root.addObject(
            "AttachConstraint",
            indices1="0",
            name="attachConstraint",
            indices2=selected_corners,
            velocityFactor="1.0",
            responseFactor="1.0",
            object1="@ball/ball_surf",
            object2="@SquareGravity",
            positionFactor="1.0",
            constraintFactor="1",
            clamp="true"
            # true if forces should be projected back from model2 to model1
            # twoWay="true",
        )

    def _place_gripper_right(self, pos2, selected_corners):
        orientation = "0 0 0 1.0"
        pos_gripper_right = pos2 + " " + orientation

        Input_2 = self.root.addChild("Input_2")
        self.Input_2 = Input_2
        # Add the Second gripper
        Input_2.addObject(
            "MechanicalObject",
            position=pos_gripper_right,
            name="DOFs2",
            template="Rigid3d",
        )

        # self.root/Input/RefModel
        RefModel = Input_2.addChild("RefModel")
        self.RefModel = RefModel
        RefModel.addObject(
            "MeshObjLoader",
            name="loader",
            filename=os.path.join(self.mesh_path, "sphere.obj"),
        )
        RefModel.addObject("Mesh", src="@loader")
        RefModel.addObject(
            "MechanicalObject", src="@loader", name="instrumentCollisionState"
        )
        RefModel.addObject("RigidMapping")

        # self.root/ball
        ball_2 = self.root.addChild("ball_2")
        self.ball_2 = ball_2
        ball_2.addObject(
            "EulerImplicitSolver",
            rayleighStiffness="0.05",
            name="ODE solver",
            rayleighMass="0.0",
        )
        ball_2.addObject(
            "CGLinearSolver",
            threshold="10e-20",
            tolerance="1e-10",
            name="linear solver",
            iterations="10",
        )
        ball_2.addObject(
            "MechanicalObject",
            translation=pos2,
            name="ballState",
            template="Rigid3d",
        )
        ball_2.addObject("UniformMass", name="mass", totalMass="0.00001")
        ball_2.addObject("LCPForceFeedback", activate="true", forceCoef="0.001")
        ball_2.addObject("UncoupledConstraintCorrection", compliance="100")

        # self.root/ball/ball_2_surf
        ball_2_surf = ball_2.addChild("ball_2_surf")
        self.ball_2_surf = ball_2_surf
        ball_2_surf.addObject(
            "MeshObjLoader",
            name="meshLoader",
            filename=os.path.join(self.mesh_path, "sphere.obj"),
        )
        ball_2_surf.addObject("Mesh", src="@meshLoader")
        ball_2_surf.addObject(
            "MechanicalObject",
            src="@meshLoader",
            name="ball2CollisionState",
            template="Vec3d",
        )
        ball_2_surf.addObject("UniformMass", totalMass="0.00001")
        ball_2_surf.addObject("UncoupledConstraintCorrection", compliance="1")
        ball_2_surf.addObject(
            "RigidMapping", input="@..", output="@ball2CollisionState"
        )

        # self.root/ball/ball_2_surf/VisualModel
        VisualModel = ball_2_surf.addChild("VisualModel")
        self.VisualModel = VisualModel
        VisualModel.addObject(
            "OglModel",
            color="1.0 0.2 0.2 1.0",
            src="@../meshLoader",
            name="ball2VisualModel",
        )
        VisualModel.addObject(
            "IdentityMapping", input="@..", output="@ball2VisualModel"
        )
        ball_2.addObject(
            "VectorSpringForceField",
            object1="@Input_2/RefModel/instrumentCollisionState",
            object2="@ball_2/ball_2_surf/ball2CollisionState",
            viscosity="0",
            stiffness="100000",
        )
        # Attach the ball to the index 440 of the cloth
        self.root.addObject(
            "AttachConstraint",
            indices1="0",
            name="attachConstraint2",
            indices2=selected_corners,
            velocityFactor="1.0",
            responseFactor="1.0",
            object1="@ball_2/ball_2_surf",
            object2="@SquareGravity",
            positionFactor="1.0",
            constraintFactor="1",
            clamp="true"
            # true if forces should be projected back from model2 to model1
            # twoWay="true",
        )

    def _place_table(self, pos_table):
        # Add the table
        Floor = self.root.addChild("Floor")
        self.Floor = Floor
        Floor.addObject(
            "MeshObjLoader",
            filename=os.path.join(self.mesh_path, "floor.obj"),
            translation=pos_table,
            scale=str(self.scale),
            # triangulate="true",
            name="loader",
        )
        Floor.addObject("MeshTopology", src="@loader")
        Floor.addObject(
            "MechanicalObject", src="@loader", name="DOFs", template="Vec3d"
        )
        Floor.addObject(
            "TriangleCollisionModel",
            moving=False,
            simulated=False,
            contactFriction="100",
        )
        Floor.addObject(
            "LineCollisionModel", moving=False, simulated=False, contactFriction="100"
        )
        Floor.addObject(
            "PointCollisionModel", moving=False, simulated=False, contactFriction="100"
        )
        Floor.addObject("OglModel", name="FloorV")

    def place_objects_in_scene(self):
        # upper_corners = [399, 419, 420, 440]
        upper_corners = [420, 440, 420, 440]
        # pos1 = self.array2str(self.position_grippers[:3])
        # pos2 = self.array2str(self.position_grippers[3:])
        pos1 = self.array2str(self.position_grippers[[0, 2, 1]])
        pos2 = self.array2str(self.position_grippers[[3, 5, 4]])

        # TODO FIX Z POSITION
        # z_pos = 5.0
        # pos_table = self.array2str(np.array([-0.2, 0.1950, 0.0])*self.scale)

        pos_table = self.array2str(np.array([-0.2, self.table_z_pos, 0.0]) * self.scale)
        pos_cloth = self.array2str(np.array([0.0, 1.0, 0.0]) * self.scale)

        self.root.addObject(
            "VisualStyle",
            displayFlags="hideBehaviorModels hideCollisionModels hideMappings hideForceFields showVisualModels hideInteractionForceFields",
        )
        self.root.addObject("FreeMotionAnimationLoop", solveVelocityConstraintFirst="0")
        self.root.addObject(
            "LCPConstraintSolver",
            mu="1.9",
            name="LCPConstraintSolver",
            maxIt="1000",
            printLog="0",
            initial_guess="false",
            multi_grid="false",
            build_lcp="true",
            tolerance="1e-6",
            # tolerance="0.001",  # headless becomes unstable
        )
        self.root.addObject("CollisionPipeline", draw="0", depth="6", verbose="0")
        self.root.addObject("BruteForceBroadPhase", name="BFB")
        self.root.addObject("BVHNarrowPhase", name="BFN")

        self.root.addObject(
            "LocalMinDistance",
            useLMDFilters="0",
            # contactDistance="01",
            # alarmDistance="03",
            contactDistance="01",
            alarmDistance="03",
            name="Proximity",
        )
        self.root.addObject(
            "CollisionResponse", name="Response", response="FrictionContactConstraint"
        )

        # Place cloth
        self._place_cloth(pos_cloth)

        # Place left and right gripper
        self._place_gripper_left(pos1, f"{upper_corners[0]} " f"{upper_corners[2]}")
        self._place_gripper_right(pos2, f"{upper_corners[1]} " f"{upper_corners[3]}")

        # Place rigid surface
        self._place_table(pos_table)

    def place_edge(self):
        # self.root/SquareGravity/Edge Mesh
        Edge_Mesh = self.SquareGravity.addChild("Edge Mesh")
        self.Edge_Mesh = Edge_Mesh
        Edge_Mesh.addObject("EdgeSetTopologyContainer", name="Container")
        Edge_Mesh.addObject("EdgeSetTopologyModifier", name="Modifier")
        Edge_Mesh.addObject(
            "EdgeSetGeometryAlgorithms",
            drawEdges="1",
            name="GeomAlgo",
            template="Vec3d",
        )
        Edge_Mesh.addObject(
            "Triangle2EdgeTopologicalMapping",
            input="@../topo",
            name="Mapping",
            output="@Container",
        )
        Edge_Mesh.addObject(
            "MeshSpringForceField",
            name="MeshSpringForceField",
            template="Vec3d",
            stiffness="100",
            damping="1.0",
        )

    def _get_info(self):
        rgb, depth = self.render()
        info = {"rgb": rgb, "depth": depth}
        return info

    def reset(self):
        Sofa.Simulation.reset(self.root)
        return None, self._get_info()

    def _perform_action(self, action):
        # action = self.trajectory[self.i]
        # print(
        #     f"Actuator positions {self.actuators[0].position[0]} {self.actuators[1].position[0]}"
        # )
        # with open(f'trajectory.txt', 'a') as f:
        #     f.write(f'Actuator positions {self.actuators[0].position[0]} {self.actuators[1].position[0]}\n')
        # action[2] += 0.01
        # action[5] += 0.01
        self.actuators[0].position[0][:3] = action[[0, 2, 1]] * self.scale
        self.actuators[1].position[0][:3] = action[[3, 5, 4]] * self.scale

    def step(self, action):
        # self.onAnimateBeginEvent(action)
        # print(f"On Sofa step")
        self._perform_action(action)
        # After action taken
        Sofa.Simulation.animate(self.root, self.dt)
        Sofa.Simulation.updateVisual(self.root)

        img, depth = self.render()

        cloth_vertices = (
            self.SquareGravity.getObject("cloth").position.array().copy() / self.scale
        )
        cloth_vertices = cloth_vertices[:, [0, 2, 1]]

        info = {
            "vertices": cloth_vertices,
            # "vertices": all_vertices,
            "rgb": img,
            "depth": depth,
        }

        # obs = self._get_obs()
        obs = img
        reward = 0
        done = False
        truncated = False

        return obs, reward, done, truncated, info

    def _get_obs(self):
        return None
        clothMesh = self.SquareGravity
        gripperPosition = np.array(
            self.ballMesh[0].getObject("ballState").position[0][0:3]
        )
        gripperVelocity = np.array(
            self.ballMesh[0].getObject("ballState").velocity[0][0:3]
        )

        gripper2Position = np.array(
            self.ballMesh[1].getObject("ballState").position[0][0:3]
        )
        gripper2Velocity = np.array(
            self.ballMesh[1].getObject("ballState").velocity[0][0:3]
        )
        (
            verticePositions,
            verticeVelocities,
            vertice_rel_positions,
        ) = ([], [], [])

        if self.edge_large:
            edge_nums = [11, 19, 34, 42, 57, 65, 80, 88]  # two points sampled per egde
        else:
            edge_nums = [15, 38, 61, 84]  # one point sampled per edge
        edgePositions, edgeVelocities = [], []

        for verticeNum in range(self.numVertices):
            if len(clothMesh.getObject("cloth").velocity) == 1:
                verticePositions.append(
                    np.array(clothMesh.getObject("cloth").position[verticeNum])
                )
                verticeVelocities.append(
                    np.array(clothMesh.getObject("cloth").velocity[0])
                )
                if self.give_relative_pos:
                    vertice_rel_positions.append(
                        np.array(verticePositions[verticeNum] - gripperPosition)
                    )
                    # vertice_rel_velocities.append(np.array(verticeVelocities[0]- gripperVelocity ))
            else:
                verticePositions.append(
                    np.array(clothMesh.getObject("cloth").position[verticeNum])
                )
                verticeVelocities.append(
                    np.array(clothMesh.getObject("cloth").velocity[verticeNum])
                )
                if self.give_relative_pos:
                    vertice_rel_positions.append(
                        np.array(verticePositions[verticeNum] - gripperPosition)
                    )

        if self.edge_observations:
            for edgeNum in edge_nums:
                if len(clothMesh.getObject("cloth").velocity) == 1:
                    edgePositions.append(
                        np.array(clothMesh.getObject("cloth").position[edgeNum])
                    )
                    edgeVelocities.append(
                        np.array(clothMesh.getObject("cloth").velocity[0])
                    )
                else:
                    edgePositions.append(
                        np.array(clothMesh.getObject("cloth").position[edgeNum])
                    )
                    edgeVelocities.append(
                        np.array(clothMesh.getObject("cloth").velocity[edgeNum])
                    )

        achieved_goal = np.concatenate(
            [
                verticePositions[0].copy(),
                verticePositions[1].copy(),
            ]
        )

        # print 'gripper Current state', gripperState
        obs = np.concatenate(
            [
                gripperPosition,
                gripperVelocity,
                gripper2Position,
                gripper2Velocity,
            ]
        )

        obs = np.append(obs, verticePositions[3].ravel(), axis=0)  # 3
        obs = np.append(obs, verticeVelocities[3].ravel(), axis=0)
        obs = np.append(obs, verticePositions[1].ravel(), axis=0)  # 3
        obs = np.append(obs, verticeVelocities[1].ravel(), axis=0)
        obs = np.append(obs, verticePositions[0].ravel(), axis=0)  # 3
        obs = np.append(obs, verticeVelocities[0].ravel(), axis=0)
        obs = np.append(obs, verticePositions[2].ravel(), axis=0)  # 3
        obs = np.append(obs, verticeVelocities[2].ravel(), axis=0)

        if self.edge_observations:
            for edgeNum in range(len(edge_nums)):
                obs = np.append(obs, edgePositions[edgeNum].ravel(), axis=0)  # 3
                obs = np.append(obs, edgeVelocities[edgeNum].ravel(), axis=0)  # 3
        if self.give_relative_pos:
            obs = np.append(obs, vertice_rel_positions[3].ravel(), axis=0)  # 3
            obs = np.append(obs, vertice_rel_positions[1].ravel(), axis=0)  # 3
            obs = np.append(obs, vertice_rel_positions[0].ravel(), axis=0)  # 3
            obs = np.append(obs, vertice_rel_positions[2].ravel(), axis=0)  # 3

        if self.explicit_policy:
            # Passing dynamic parameters in an explicit policy
            clothMass = (
                self.node.getChild("SquareGravity")
                .getObject("UniformMass")
                .findData("mass")
                .getValue(0)
            )
            obs = np.append(obs, clothMass)
        else:
            None

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
        }

    def render(self):
        img = None
        # Sofa.Gui.GUIManager.SaveScreenshot(self.tmp_file)
        # img = Image.open(self.tmp_file)
        depth = None
        return np.asarray(img), depth

class ClothPickers(Sofa.Core.Controller):
    def __init__(
        self,
        cloth,
        actuators,
        dt,
        scale,
        trajectory,
        pre_traj_length,
        compute_metrics_fn,
        targets,
        target_it,
        target_name,
        select_targets,
        *args,
        **kwargs,
    ):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.actuators = actuators
        self.cloth = cloth
        #
        # z_middle = z_start + 0.056
        # z_end = z_start - 0.514
        # z_middle = fling_height
        # z_end = grasp_height

        self.scale = scale
        self.trajectory = trajectory
        self.pre_traj_length = pre_traj_length
        self.compute_metrics_fn = compute_metrics_fn

        self.target_name = target_name
        self.targets = targets
        self.target_it = target_it
        self.select_targets = select_targets

        self.i = 0
        print(
            f"Actuator positions {self.actuators[0].position[0]} {self.actuators[1].position[0]}"
        )

    def onAnimateBeginEvent(self, event):
        # This method is internally called by Sofa at each step

        cloth_vertices = (
            self.cloth.getObject("cloth").position.array().copy() / self.scale
        )

        # print(f"Cloth vertices shape={cloth_vertices.shape}")
        # elev = 8
        # azim = 13
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(cloth_vertices[:, 0], cloth_vertices[:, 2], cloth_vertices[:, 1],
        #            marker='*', color='r', label='Sim')
        # ax.view_init(elev=elev, azim=azim)
        # plt.savefig('sofa_cloth.png')
        # plt.close()

        output_path = "sofa/None/"
        if not os.path.exists(output_path):
            print(f"Creating path={output_path}")
            os.makedirs(output_path)

        info = {"vertices": cloth_vertices}

        if self.i < len(self.trajectory):
            action = self.trajectory[self.i]
            # print(
            #     f"Actuator positions {self.actuators[0].position[0]} {self.actuators[1].position[0]}"
            # )
            with open("trajectory.txt", "a") as f:
                f.write(
                    f"Actuator positions {self.actuators[0].position[0]} {self.actuators[1].position[0]}\n"
                )
            self.actuators[0].position[0][:3] = action[[0, 2, 1]] * self.scale
            self.actuators[1].position[0][:3] = action[[3, 5, 4]] * self.scale

            if self.i >= self.pre_traj_length:
                j = self.i - self.pre_traj_length
                print(f"Value of it j={j}, i={self.i}")

                if self.target_name != "None":
                    # For MuJoCo there is a difference of start in the movement of the trajectory of 10 steps
                    # So we define the skip of frames
                    skip_frames = 8

                    if j > skip_frames:  # Compare to 131 to check the rotation
                        cost_fabric, cost_contact, diff_z = self.compute_metrics_fn(
                            self.targets,
                            self.target_it,
                            self.target_name,
                            j,
                            info,
                            "sofa",
                            self.select_targets,
                        )

                # plot_sim_and_target(torch.from_numpy(cloth_vertices)[None, :, [0, 2, 1]], None, j,
                #                     output_path)
                # plot_sim_and_target(torch.from_numpy(cloth_vertices)[None, :, [0, 2, 1]], None, j,
                #                     output_path, elev=0, azim=90)

            self.i += 1
