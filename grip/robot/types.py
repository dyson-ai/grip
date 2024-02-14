from dataclasses import dataclass
import numpy as np


@dataclass
class BulletDynamicsInfo:
    """
    Information pertinent to the dynamics of a rigid body link.
    More info at: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#

    """

    mass: float  # mass in kg
    lateral_friction: float  # friction coefficient
    local_inertia_diagonal: (
        np.ndarray
    )  # local inertia diagonal. Note that links and base are centered around the center of mass and aligned with the principal axes of inertia.
    local_inertial_pos: (
        np.ndarray
    )  # position of inertial frame in local coordinates of the joint frame
    local_inertial_orn: (
        np.ndarray
    )  # orientation of inertial frame in local coordinates of joint frame
    resolution: float  # coefficient of restitution
    rolling_friction: float  # rolling friction coefficient orthogonal to contact normal
    spinning_friction: float  # spinning friction coefficient around contact normal
    contact_damping: float  # -1 if not available. damping of contact constraints.
    contact_stiffness: float  # -1 if not available. stiffness of contact constraints.
    body_type: int  # 1=rigid body, 2 = multi body, 3 = soft body
    collision_margin: float  # advanced/internal/unsupported info. collision margin of the collision shape. collision margins depend on the shape type, it is not consistent.

    def __post_init__(self):
        self.local_inertia_diagonal = np.array(self.local_inertia_diagonal)
        self.local_inertial_pos = np.array(self.local_inertial_pos)
        self.local_inertial_orn = np.array(self.local_inertial_orn)


@dataclass
class BulletJointInfo:
    """
    Information pertinent to the joints of a rigid body linkage.
    More info at: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#

    """

    joint_index: int  # the same joint index as the input parameter
    joint_name: str  # the name of the joint, as specified in the URDF (or SDF etc) file
    joint_type: int  # type of the joint, this also implies the number of position and velocity variables. JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED. See the section on Base, Joint and Links for more details.
    q_index: (
        int  # the first position index in the positional state variables for this body
    )
    u_index: (
        int  # the first velocity index in the velocity state variables for this body
    )
    flags: int  # reserved
    joint_damping: float  # the joint damping value, as specified in the URDF file
    joint_friction: float  # the joint friction value, as specified in the URDF file
    joint_lower_limit: (
        np.ndarray
    )  # Positional lower limit for slider and revolute (hinge) joints, as specified in the URDF file.
    joint_upper_limit: (
        np.ndarray
    )  # Positional upper limit for slider and revolute joints, as specified in the URDF file. Values ignored in case upper limit <lower limit
    joint_max_force: float  # Maximum force specified in URDF (possibly other file formats) Note that this value is not automatically used. You can use maxForce in 'setJointMotorControl2'
    joint_max_velocity: float  # Maximum velocity specified in URDF. Note that the maximum velocity is not used in actual motor control commands at the moment.
    link_name: str  # the name of the link, as specified in the URDF (or SDF etc.) file
    joint_axis: np.ndarray  # joint axis in local frame (ignored for JOINT_FIXED)
    parent_frame_pos: np.ndarray  # joint position in parent frame
    parent_frame_orn: (
        np.ndarray
    )  # joint orientation in parent frame (quaternion x,y,z,w)
    parent_index: int  # parent link index, -1 for base

    def __post_init__(self):
        self.joint_lower_limit = np.array(self.joint_lower_limit)
        self.joint_upper_limit = np.array(self.joint_upper_limit)
        self.parent_frame_pos = np.array(self.parent_frame_pos)
        self.parent_frame_orn = np.array(self.parent_frame_orn)


@dataclass
class VisualShapeData:
    """
    Visual shape data fields. The return of p.getVisualShapeData(...)
    more info at: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
    """

    obj_unique_id: int  # object unique id, same as the input
    link_id: int  # link index or -1 for the base
    vis_geom_type: int  # visual geometry type (TBD)
    dims: (
        np.ndarray
    )  # vec3, list of 3 floats, dimensions (size, local scale) of the geometry
    mesh_asset_fname: str  # path to the triangle mesh, if any. Typically relative to the URDF, SDF or MJCF file location, but could be absolute.
    local_vis_frame_pos: (
        np.ndarray
    )  # position of local visual frame, relative to link/joint frame
    local_vis_frame_orn: (
        np.ndarray
    )  # orientation of local visual frame relative to link/joint frame
    rgba_colour: np.ndarray  # URDF color (if any specified) in red/green/blue/alpha
    # texture_uid: int  # (field only exists if using VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS flags)Texture unique id of the shape, or -1 if none

    def __post_init__(self):
        self.dims = np.array(self.dims)
        self.local_vis_frame_pos = np.array(self.local_vis_frame_pos)
        self.local_vis_frame_orn = np.array(self.local_vis_frame_orn)
        self.rgba_colour = np.array(self.rgba_colour)


@dataclass
class BulletContactInfo:
    """
    Contact information between two rigid body links
    More info at: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#

    """

    contact_flag: int  # reserved
    body_uid_a: str  # body unique id of body A
    body_uid_b: int  # body unique id of body B
    link_id_a: int  # link index of body A, -1 for base
    link_id_b: int  # link index of body B, -1 for base
    position_on_a: np.ndarray  # contact 3D position on A, in Cartesian world coordinate
    position_on_b: (
        np.ndarray
    )  # contact 3D position on B, in Cartesian world coordinates
    contact_normal_on_b: np.ndarray  # contact normal on B, pointing towards A
    contact_distance: (
        float  # contact distance, positive for separation, negative for penetration
    )
    normal_force: float  # normal force applied during the last 'world.step()'
    lateral_friction1: (
        float  # lateral friction force in the lateralFrictionDir1 direction
    )
    lateral_friction_dir1: np.ndarray  # first lateral friction direction
    lateral_friction2: (
        float  # lateral friction force in the lateralFrictionDir2 direction
    )
    lateral_friction_dir2: np.ndarray  # second lateral friction direction

    def __post_init__(self):
        self.position_on_a = np.array(self.position_on_a)
        self.position_on_b = np.array(self.position_on_b)
        self.contact_normal_on_b = np.array(self.contact_normal_on_b)
        self.lateral_friction_dir1 = np.array(self.lateral_friction_dir1)
        self.lateral_friction_dir2 = np.array(self.lateral_friction_dir2)
