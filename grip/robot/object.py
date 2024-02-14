import pybullet as p
import numpy as np
from typing import Union, List, Tuple, Dict, TypedDict, Optional

from ..io import log
from ..math import quaternion_error
from .entity import Entity
from .world import BulletWorld
import uuid, os

ObjectState = TypedDict(
    "State", {"pose": Tuple[np.ndarray, np.ndarray], "entity": Entity}
)


class BulletObject(Entity):
    """
    The BulletObject represents an interface to a Bullet rigid body.
    An object of this class exposes several utilities for manipulating rigid bodies.

    Args:
        world (grip.robot.BulletWorld): parent world where the object where the object belongs to.
        urdf_file (str): urdf file path where this object should be loaded from
        id (int, optional): unique identifier of an already existing object (potentially an object already existing in an external physics world server)
            There is no need to pass urdf_file path if id is passed as parameter
        position (numpy.ndarray): shape-(3,) position in the world where the object to be located.
        orientation (numpy.ndarray): shape-(4,) orientation quaternion in the format [x, y, z, w] (JPL format).
        fixed (bool): whether or not the object is anchored/fixed to the world (a fixed object will not budge).
        scale (float): global scaling of the object that allows one to change the object size.
        compliant_dyn (bool): contacts may be less rigid by setting this flag to true (compliant/springy contacts)
    """

    def __init__(self, **kwargs):
        self.uuid = str(uuid.uuid4())

        self._parent_world = kwargs.get("world", None)
        self._urdf_file = kwargs.get("urdf_file", None)

        self._position = kwargs.get("position", [0, 0, -0.3])
        self._orientation = kwargs.get("orientation", [0, 0, 0, 1])
        self._fixed_base = kwargs.get("fixed", True)
        self._scale = kwargs.get("scale", 1.0)
        self._compliant_dyn = kwargs.get("compliant_dyn", False)

        self.phys_id = 0 if self._parent_world is None else self._parent_world.id

        self._mesh = kwargs.get("mesh", None)
        self._mesh_mass = kwargs.get("mesh_mass", 0.2) if not self._fixed_base else 0
        self._mesh_colour = kwargs.get("mesh_colour", None)

        self._flip_normals = kwargs.get("flip_normals", True)

        self._is_concave = kwargs.get("is_concave", False)
        self._force_concave = kwargs.get("force_concave", False)
        self._disable_collision = kwargs.get("disable_collision", False)
        self._with_texture = kwargs.get("with_texture", True)

        self._id = kwargs.get("id", None)

        self._texture_path = None

        flags = 0

        if self._force_concave:
            flags |= p.GEOM_FORCE_CONCAVE_TRIMESH

        # p.URDF_USE_MATERIAL_COLORS_FROM_MTL

        self._vid = None

        if self._id is None and self._urdf_file is not None:
            self._id = p.loadURDF(
                self._urdf_file,
                self._position,
                self._orientation,
                useFixedBase=self._fixed_base,
                globalScaling=self._scale,
                flags=flags,
                physicsClientId=self.phys_id,
            )
        elif self._mesh is not None:
            self._id, self._vid, self._cid = self._init_from_mesh(self._mesh)
            # disable rendering during creation.

        if self._compliant_dyn:
            # p.changeDynamics(self._id, -1, lateralFriction=1.0, physicsClientId=self.phys_id)
            p.changeDynamics(
                self._id,
                -1,
                contactStiffness=5000,
                contactDamping=500,
                physicsClientId=self.phys_id,
            )

        if self._parent_world is not None:
            self.texture_randomiser = kwargs.get(
                "texture_randomiser", self._parent_world.texture_randomiser
            )
        else:
            self.texture_randomiser = kwargs.get("texture_randomiser", None)

        self._cid = None
        self._base_cid = None
        self._max_force = 500

        self.cached_pose = self.pose

        self.link_dict = self._get_link_dict()

    def _get_mesh_data(self, mesh, flip_normals=True, guiding_normal=[0, 0, 1]):
        import trimesh

        n_verts = len(mesh.vertices)
        uvs = [list(np.random.rand(2)) for i in range(n_verts)]
        try:
            if (
                isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals)
                and len(mesh.visual.uv) > 0
            ):
                uvs = mesh.visual.uv
            else:
                unwrapped_mesh = mesh.unwrap()
                uvs = unwrapped_mesh.visual.uv

        except Exception as e:
            log.warning("Unwrap unavailable: {}".format(e))

        vertices = mesh.vertices

        faces = mesh.faces.reshape(
            -1,
        )

        normals = np.array(mesh.vertex_normals)

        mean_normal = np.mean(normals, axis=0)

        if flip_normals and np.dot(mean_normal, guiding_normal) < 0:
            normals *= -1
            faces = faces[::-1]

        return vertices, faces, normals, uvs

    def _init_from_mesh(self, mesh):
        import trimesh

        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.phys_id)
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.phys_id)
        # disable tinyrenderer, software (CPU) renderer, we don't use it here

        shift = [0, 0.0, 0]
        meshScale = [self._scale, self._scale, self._scale]

        log.info("Mesh kind: {}".format(mesh.visual.kind))

        vertices, faces, normals, uvs = self._get_mesh_data(
            mesh, flip_normals=self._flip_normals
        )

        # normals = -normals
        # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)

        colour = None

        self._tid = None

        if self._mesh_colour is None and isinstance(
            mesh.visual, trimesh.visual.color.ColorVisuals
        ):
            try:
                colour = trimesh.visual.color.to_float(mesh.visual.vertex_colors)
                log.info("Has colours look: {}".format(colour))
            except:
                log.info("Failed to retrieve colour")

        elif (
            self._mesh_colour is None
            and self._with_texture
            and isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals)
        ):
            colour = trimesh.visual.color.to_float(mesh.visual.to_color().vertex_colors)

            self._texture_path = "/tmp/grip_tex{}.jpg".format(self.uuid)
            mesh.visual.material.image.save(self._texture_path)
            self._parent_world.add_texture(self._texture_path)
            self._tid = self._parent_world.texture(self._texture_path)

        if colour is None:
            colour = (
                [0.8, 0.8, 0.2, 1.0] if self._mesh_colour is None else self._mesh_colour
            )

        vid = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            rgbaColor=colour,
            specularColor=[0.7, 0.4, 0],
            visualFramePosition=shift,
            meshScale=meshScale,
            vertices=vertices,
            indices=faces,
            normals=normals,
            uvs=uvs,
            physicsClientId=self.phys_id,
        )
        cid = -1

        if not self._disable_collision:
            if self._is_concave:
                cid = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    vertices=vertices,
                    collisionFramePosition=shift,
                    # flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                    meshScale=meshScale,
                    physicsClientId=self.phys_id,
                )
            else:
                cid = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    vertices=vertices,
                    collisionFramePosition=shift,
                    # flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                    meshScale=meshScale,
                    physicsClientId=self.phys_id,
                )

        bid = p.createMultiBody(
            baseMass=self._mesh_mass,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=cid,
            baseVisualShapeIndex=vid,
            basePosition=[0, 0, 0],
            useMaximalCoordinates=False,
            physicsClientId=self.phys_id,
        )

        # shape_data = p.getVisualShapeData(bid, flags=p.VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS, physicsClientId=self.phys_id)

        if self._tid is not None:
            p.changeVisualShape(bid, -1, textureUniqueId=self._tid)

        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.phys_id)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self.phys_id)

        return bid, vid, cid

    def __del__(self):
        if self._texture_path is not None:
            log.info("Clearing resources Bullet object {}".format(self.uuid))
            os.system("rm -rf {}".format(self._texture_path))

    def reset_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """
        Sets position and orientation of this object

        Args:
            pos (numpy.ndarray): shape-(3,) array three dimensional position
            quat (numpy.ndarray): shape-(4,) orientation quaternion following JPL format [x, y, z, w], where w is the real part.

        """

        if self._base_cid is not None:
            p.changeConstraint(self._base_cid, pos, quat, maxForce=self._max_force)
        else:
            p.resetBasePositionAndOrientation(
                self._id, pos, quat, physicsClientId=self.phys_id
            )

    def set_pose(self, pos, quat) -> None:
        """
        Sets position and orientation of this object.
        If the object has a constraint, then the constraint position is changed
        Otherwise, the base pose is reset to the chosen pose.

        Args:
            pos (numpy.ndarray): shape-(3,) array three dimensional position
            quat (numpy.ndarray): shape-(4,) orientation quaternion following JPL format [x, y, z, w], where w is the real part.

        """
        if self._cid is None:
            self.reset_pose(pos, quat)

        else:
            p.changeConstraint(self._cid, pos, quat, maxForce=500)

    def set_colour(self, rgba: List[float]) -> None:
        """
        Sets global color of this object

        Args:
            rgba (list[float]): rgba colour of this object

        """
        if self._vid is None:
            self._vid = -1
        p.changeVisualShape(
            self._id, -1, self._vid, rgbaColor=rgba, physicsClientId=self.phys_id
        )

    def set_friction(self, lateral_friction: float) -> None:
        """Sets object lateral (linear) contact friction coefficient

        Args:
            friction (float): lateral (linear) contact friction coefficient
        """

        p.changeDynamics(self.id, -1, lateralFriction=lateral_friction)

    def set_mass(self, mass: float) -> None:
        """Sets object mass in Kg

        Args:
            mass (float): object's mass in Kg
        """
        p.changeDynamics(self.id, -1, mass=mass)

    def get_dimensions(self) -> np.ndarray:
        """
        Gets the dimensions of this object.

        Returns:
            (numpy.ndarray): depends on geometry type: for GEOM_BOX: extents,
            for GEOM_SPHERE dimensions[0] = radius,
            for GEOM_CAPSULE and GEOM_CYLINDER, dimensions[0] = height (length),
            dimensions[1] = radius.
            For GEOM_MESH, dimensions is the scaling factor.


        """
        data = p.getCollisionShapeData(self.id, -1, physicsClientId=self.phys_id)[0]

        return np.array(data[3])

    def random_texture(self) -> None:
        """
        Randomises texture of this object based on pre-loaded world textures

        """

        if self.texture_randomiser is not None:
            self.texture_randomiser.randomise(self)

    def load_texture(self, texture: Union[List[float], str]) -> None:
        """
        Loads and sets the texture for this object. If texture is a rgba colour, then the colour of this object is set instead.


        Args:

        texture (Union[List[float], str]): rgba colour or string file path to a texture image file.

        """

        assert type(texture) in [list, str, np.str_], "Please provide a valid texture"
        if isinstance(texture, list):
            rgbaColor = texture[:]
            if self._vid is None:
                self._vid = -1
            self.set_colour(rgbaColor)
        elif isinstance(texture, str) or isinstance(texture, np.str_):
            texture_fn = texture
            assert os.path.isfile(texture_fn), "Texture file {} doesn't exit".format(
                texture_fn
            )
            texture_id = p.loadTexture(
                os.path.abspath(texture_fn), physicsClientId=self.phys_id
            )
            p.changeVisualShape(
                objectUniqueId=self._id,
                linkIndex=-1,
                textureUniqueId=texture_id,
                physicsClientId=self.phys_id,
            )

    def remove(self) -> None:
        """
        Removes this object from the world.

        """

        p.removeBody(self._id, physicsClientId=self.phys_id)

    @property
    def id(self):
        """
        int: this object's unique identifier
        """
        return self._id

    @property
    def pose(self):
        """
        Tuple[numpy.ndarray, numpy.ndarray]: this object's pose comprised of three-dimentional position and orientation quaternion of this object.


        """

        position, orientation = p.getBasePositionAndOrientation(
            self._id, physicsClientId=self.phys_id
        )

        return np.array(position), np.array(orientation)

    @property
    def base_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the current base velocity of this robot, as a linear and angular velocity in cartesian world coordinates.

        Returns:
            (Tuple[numpy.ndarray, numpy.ndarray]): tuple containing a linear shape-(3,) and angular shape-(3,) velocities.
        """
        lin, ang = p.getBaseVelocity(self.id, physicsClientId=self.phys_id)

        return np.asarray(lin), np.asarray(ang)

    @property
    def shape_data(self):
        return p.getVisualShapeData(self.id, -1, physicsClientId=self.phys_id)

    def contact_points(self, entity: Entity) -> List[
        Tuple[
            int,
            int,
            int,
            int,
            int,
            List[float],
            List[float],
            List[float],
            float,
            float,
            float,
            List[float],
            float,
            List[float],
        ]
    ]:
        """
        Returns contact points between this object and another entity given as parameter


        Args:

        entity (grip.robot.Entity): entity we would like to query contact existance

        Returns:
            (List[Tuple[int, int, int, int, int, List[float], List[float], List[float], float, float, float, List[float], float, List[float]]]):
                returns list of bullet contact points
                for more details see: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.cb0co8y2vuvc

        """

        contact_points = p.getContactPoints(
            entity.id, self.id, physicsClientId=self.phys_id
        )

        return contact_points

    def is_contacting(self, entity: Entity) -> bool:
        """
        Simple binary check if this object is contacting another entity or not

        Args:
            entity (grip.robot.Entity):
                entity we would like to query contact existance

        Returns:
            bool: whether or not this object is contacting the given entity

        """

        contacts = self.contact_points(entity)

        return len(contacts) > 0

    def state(self) -> ObjectState:
        """
        Returns the current state of this object


        Returns:
            (ObjectState): dictionary containing the current state of this object.

        """

        state = dict()

        state["pose"] = self.pose
        state["entity"] = self

        return state

    def record_pose(self) -> None:
        """This stores the pose of object when called.  At a later time youc an called self.moved() to see if the object has moved from the stored pose"""
        self.cached_pose = self.pose
        return self.cached_pose

    def moved(self, pos_threshold: float = 0, ang_threshold: float = 0) -> bool:
        """Call this function to check if the object has moved since the last time record_pose() was called

        Args:
            pos_threshold (float, optional): the maximum distance in meters the object is allowed to have moved and still considered static. Defaults to 0.
            ang_threshold (float, optional): the maximum angular distance in radians the object is allowed to have moved and still considered static. Defaults to 0.

        Returns:
            bool: _description_
        """
        return (
            np.linalg.norm(self.cached_pose[0] - self.pose[0]) > pos_threshold
            and quaternion_error(self.cached_pose[1], self.pose[1]) > ang_threshold
        )

    def reset(self) -> None:
        """
        Resets this object pose to the original position and orientation passed in its contructor during its instantiation

        """

        self.reset_pose(self._position, self._orientation)

    def aabb(self, expansion: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        The returns the axis-aligned bounding box of this object


        Returns:
            (Tuple[numpy.ndarray, numpy.ndarray]): a tuple of 3-d vectors (aabb_min, aabb_max) representing the minimum and maximum bounds  in world frame.

        """

        aabb = p.getAABB(self.id, -1, physicsClientId=self.phys_id)

        aabb_min = aabb[0]
        aabb_max = aabb[1]

        return np.array(aabb_min) - expansion, np.array(aabb_max) + expansion

    def create_constraint(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        entity: Entity = None,
        link_name: str = None,
    ) -> None:
        """
        Creates a rigid constraint to which this object is attached to.

        Args:
            position (numpy.ndarray): position of the constraint
            orientation (numpy.ndarray): orientation of the constraint
            entity (grip.robot.Entity, optional): if passed as argument, then position and orientation is relative to a chosen link name of the child entity
            link_name (str, optional): chosen link name relative to which this object will be attached to. Defaults to None, meaning base_link of the child entity.

        """
        if self._cid is not None:
            return

        child_body = -1 if entity is None else entity.id

        child_link = -1
        if entity is not None and link_name is not None:
            child_link = entity.get_link_id(link_name)

        self._cid = p.createConstraint(
            self._id,
            -1,
            child_body,
            child_link,
            p.JOINT_FIXED,
            [0, 0, 0],  # jointAxis
            [0, 0, 0],  # parentFramePosition
            position,  # childFramePosition
            [0, 0, 0, 1],  # parentFrameOrientation
            orientation,  # childFrameOrientation
            physicsClientId=self.phys_id,
        )

    def create_base_constraint(self):
        if self._base_cid is not None:
            return

        self._position_des = [0, 0, 0]
        self._orientation_des = [0, 0, 0, 1]
        ho = p.getQuaternionFromEuler([0, 0, 0])
        self._base_cid = p.createConstraint(
            self._id,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0.0, 0, 0],
            [0, 0, 0],  # [0.500000, 0.300006, 0.700000],
            ho,
            physicsClientId=self.phys_id,
        )

    def remove_base_constraint(self):
        if self._base_cid is None:
            return

        p.removeConstraint(self._base_cid, physicsClientId=self.phys_id)

        self._base_cid = None

    def disable_object_collision(self, object_entity: Entity) -> None:
        """
        Disables collision between this object and another object_entity

        Args:
            object_entity (grip.robot.Entity): entity with which this object will no longer collide with.
        """
        p.setCollisionFilterPair(
            self.id, object_entity.id, -1, -1, 0, physicsClientId=self.phys_id
        )

    def disable_collision(self) -> None:
        """
        Disables collision between this object and everything else

        """
        p.setCollisionFilterGroupMask(self.id, -1, 0, 0, physicsClientId=self.phys_id)

    def remove_constraint(self) -> None:
        """
        Disables collision between this object and everything else

        """
        if self._cid is not None:
            p.removeConstraint(self._cid, physicsClientId=self.phys_id)

        self._cid = None

    def get_link_ids(self, link_names: Optional[list[str]] = None) -> List[int]:
        """
        Gets the list of link IDs of this robot.

        Args:
            link_names (list[str], optional): optional list of link names. If not passed method returns all link_ids of this kinematic chain
        Returns:
            (List[int]): list of unique identifiers the links belonging to this robot, as loaded by the physics engine.
        """

        if link_names:
            # link ids for selected link names
            return [self.link_dict[lname] for lname in link_names]

        return [lid for _, lid in self.link_dict.items()]

    def _get_link_dict(self) -> Dict[str, int]:
        """
        Gets a dictionary mapping link names to link ids for this object

        Returns:
            (Dict[str, int]): a dictionary mapping link names to corresponding link ids
        """
        link_dict = {}

        for i in range(p.getNumJoints(self.id, physicsClientId=self.phys_id)):
            joint_info = p.getJointInfo(self.id, i, physicsClientId=self.phys_id)

            link_name = joint_info[12].decode("utf-8")
            link_dict[link_name] = i

        # explicitly including base link
        link_dict["-1base"] = -1
        return link_dict


def create_box(
    position: np.ndarray,
    orientation: np.ndarray,
    dimensions_half: np.ndarray,
    world: BulletWorld,
    mass: float = 0.0,
    colour: List[float] = [0.5, 0.5, 0.5, 1.0],
    collision_on: bool = True,
) -> BulletObject:
    """
    Creates a box-shaped BulletObject

    Args:
        position (numpy.ndarray): position of the object
        orientation (numpy.ndarray): orientation of the object
        world (grip.robot.BulletWorld): physics world the created object should belong to.
        mass (float, optional): mass of the object, default is 0.0, meaning it's a static fixed object. Set mass > 0 for dynamic object.
        colour (list[float]): rgba colour of the object.
        collision_on (bool): whether or not this object should have a collision shape.
    """

    shape_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=dimensions_half,
        rgbaColor=colour,
        physicsClientId=world.id,
    )

    box_id = -1
    if collision_on:
        collision_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=dimensions_half, physicsClientId=world.id
        )
        box_id = p.createMultiBody(
            mass,
            collision_id,
            shape_id,
            position,
            orientation,
            physicsClientId=world.id,
        )
    else:
        box_id = p.createMultiBody(
            mass, -1, shape_id, position, orientation, physicsClientId=world.id
        )

    obj = BulletObject(
        id=box_id,
        world=world,
        position=position,
        orientation=orientation,
        mesh_colour=colour,
    )
    obj._vid = shape_id

    return obj


def load_sdf(
    filename: str, world: BulletWorld = None, make_z_up: bool = False
) -> List[BulletObject]:
    """Loads a SDF file, returning the list of objects defined within it.
    Which can be later manipulated individually as needed.

    Args:
        filename (str): a relative or absolute path to the SDF file on the file system of the physics server
        world (BulletWorld, optional): world where the sdf should be spawned at. Defaults to None, meaning the default world.

    Returns:
        List[BulletObject]: list of objects loaded from sdf file
    """

    uid_list = p.loadSDF(filename)

    # SDFs may be defined with y-up convention.
    # The default coordinate frame in Bullet is z-up
    # Therefore, if the z-up convention is preferred, make_z_up = True will rotate the entities accordingly.
    if make_z_up:
        for ob_id in uid_list:
            p.resetBasePositionAndOrientation(
                ob_id,
                [0.0, 0.0, 0.0],
                [np.sin(np.pi * 0.25), 0, 0, np.cos(np.pi * 0.25)],
            )

    object_list = [BulletObject(id=uid, world=world) for uid in uid_list]

    return object_list
