import pybullet as p
import numpy as np
from ..math import (
    quaternion2rotation_matrix,
    position_from_matrix,
    quaternion_from_matrix,
)


class BulletSliders:
    def __init__(
        self, phys_id, names, min_vals, max_vals, dft_vals, map_func=lambda _: _
    ):
        """
        Creates a Bullet3 slider GUI components

        Parameters:
            phys_id: <int> physics engine ID
            names: [<str>] labels for each slider
            min_vals: lower limits
            max_vals: upper limits
            dft_vals: default initial values
            map_func: optional map function with signature f(x) -> y used to convert the readings returned by read()
                      default is identity

        """

        assert (
            len(names) == len(min_vals)
            and len(names) == len(max_vals)
            and len(names) == len(dft_vals)
        )

        self.phys_id = phys_id
        self.names = names
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.dft_vals = dft_vals

        self.size = len(self.names)

        self.map_func = map_func

        self.sliders = []

        self.reset()

        self.prev_values = self.read(raw=True)

    def has_changed(self):
        current_values = self.read(raw=True)
        has_changed = not np.isclose(
            np.linalg.norm(self.prev_values - current_values), 0.0
        )

        self.prev_values = current_values

        return has_changed

    def read(self, raw: bool = False):
        values = np.zeros(self.size)
        for i, slider in enumerate(self.sliders):
            values[i] = p.readUserDebugParameter(slider, physicsClientId=self.phys_id)

        return values if raw else self.map_func(values)

    def reset(self):
        self.remove()

        self.sliders = [
            p.addUserDebugParameter(name, ll, ul, dft, physicsClientId=self.phys_id)
            for name, ll, ul, dft in zip(
                self.names, self.min_vals, self.max_vals, self.dft_vals
            )
        ]

    def remove(self):
        for slider in self.sliders:
            p.removeUserDebugItem(slider, physicsClientId=self.phys_id)


class BulletButton(BulletSliders):
    def __init__(self, phys_id: int, name: str):
        """
        Creates a Bullet3 button GUI component

        Args:
            phys_id: physics engine identifier (the id of a BulletWorld)
            name: name and label of this button
        """

        super().__init__(phys_id, [name], [1], [0], [0])

    def was_pressed(self) -> bool:
        """
        Checks if this button was pressed.

        Returns:
            (bool): whether or not this button was pressed since previously checked.
        """

        return self.has_changed()


def add_debug_frame(pos, quat, length=0.2, lw=2, duration=0.2, pcid=0, item_ids=None):
    rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3).transpose()
    toX = pos + length * rot_mat[0, :]
    toY = pos + length * rot_mat[1, :]
    toZ = pos + length * rot_mat[2, :]
    item_ids = [-1, -1, -1] if not item_ids else item_ids
    item_ids[0] = p.addUserDebugLine(
        pos,
        toX,
        [1, 0, 0],
        lw,
        duration,
        physicsClientId=pcid,
        replaceItemUniqueId=item_ids[0],
    )
    item_ids[1] = p.addUserDebugLine(
        pos,
        toY,
        [0, 1, 0],
        lw,
        duration,
        physicsClientId=pcid,
        replaceItemUniqueId=item_ids[1],
    )
    item_ids[2] = p.addUserDebugLine(
        pos,
        toZ,
        [0, 0, 1],
        lw,
        duration,
        physicsClientId=pcid,
        replaceItemUniqueId=item_ids[2],
    )
    return item_ids


addDebugFrame = add_debug_frame


def add_debug_SE3(
    se3_pose, length: float = 0.1, lw=2, duration: float = 0.0, pcid: int = 0
) -> None:
    """wrapper of add_debug_frame that accepts a SE3 type

    Args:
        se3_pose (_type_): the pose of the frame
        length (float, optional): length of the coordinate frame lines. Defaults to 0.2.
        lw (int, optional): _description_. Defaults to 2.
        duration (float, optional): _description_. Defaults to 0.2.
        pcid (int, optional): pybullet client id. Defaults to 0.
    """
    add_debug_frame(
        se3_pose.translation(),
        se3_pose.quat(),
        length=length,
        lw=lw,
        duration=duration,
        pcid=pcid,
    )


def add_debug_4x4(
    pose_4x4, length: float = 0.1, lw=2, duration: float = 0.0, pcid: int = 0
) -> None:
    """wrapper of add_debug_frame that accepts a SE3 type

    Args:
        pose_4x4 (_type_): the pose of the frame
        length (float, optional): length of the coordinate frame lines. Defaults to 0.2.
        lw (int, optional): _description_. Defaults to 2.
        duration (float, optional): _description_. Defaults to 0.2.
        pcid (int, optional): pybullet client id. Defaults to 0.
    """
    add_debug_frame(
        position_from_matrix(pose_4x4),
        quaternion_from_matrix(pose_4x4),
        length=length,
        lw=lw,
        duration=duration,
        pcid=pcid,
    )


def visualizePcd(pcd, ratio=0.1, duration=10):
    display_freq = int(1 / ratio)
    i = 0
    for p in np.array(pcd.points):
        if (i % display_freq) == 0:
            addDebugFrame(p, [0, 0, 0, 1], length=0.001, duration=10)
        i += 1


visualise_pcd = visualizePcd


def visualize_volume(box, rgb=[1, 0, 0]):
    x_min, x_max = box[0]
    y_min, y_max = box[1]
    z_min, z_max = box[2]

    p1 = [x_min, y_max, z_min]
    p2 = [x_max, y_max, z_min]
    p3 = [x_max, y_min, z_min]
    p4 = [x_min, y_min, z_min]
    p5 = [x_min, y_max, z_max]
    p6 = [x_max, y_max, z_max]
    p7 = [x_max, y_min, z_max]
    p8 = [x_min, y_min, z_max]

    p.addUserDebugLine(p1, p2, rgb)
    p.addUserDebugLine(p1, p5, rgb)
    p.addUserDebugLine(p2, p3, rgb)
    p.addUserDebugLine(p2, p6, rgb)
    p.addUserDebugLine(p3, p4, rgb)
    p.addUserDebugLine(p3, p7, rgb)
    p.addUserDebugLine(p4, p1, rgb)
    p.addUserDebugLine(p4, p8, rgb)
    p.addUserDebugLine(p5, p6, rgb)
    p.addUserDebugLine(p6, p7, rgb)
    p.addUserDebugLine(p7, p8, rgb)
    p.addUserDebugLine(p8, p5, rgb)
    return


def add_text(
    pos, text, text_size=1.2, color=[0, 0, 0], duration=0.0, cid=0, item_id=-1
):
    item_id = p.addUserDebugText(
        text=text,
        textPosition=pos,
        textSize=text_size,
        textColorRGB=color,
        lifeTime=duration,
        replaceItemUniqueId=item_id,
        physicsClientId=cid,
    )

    return item_id


addText = add_text


def draw_frustum(
    cam_pos, cam_ori, hfov, vfov, near, far, ratio, duration=0.0, cid=0, item_ids=None
):
    rot_mat = quaternion2rotation_matrix(cam_ori)
    pos = np.array(cam_pos)

    camRight = rot_mat[:3, 0]
    camUp = rot_mat[:3, 1]
    camForward = -rot_mat[:3, 2]

    pos[0]
    pos[1]
    pos[2]

    nearCenter = pos - camForward * near
    farCenter = pos - camForward * far

    nearHeight = 2 * np.tan(vfov / 2) * near
    farHeight = 2 * np.tan(vfov / 2) * far
    nearWidth = nearHeight * ratio  # 2 * np.tan(hfov / 2) * near
    farWidth = farHeight * ratio  # 2 * np.tan(hfov / 2) * far

    farTopLeft = farCenter + camUp * (farHeight * 0.5) - camRight * (farWidth * 0.5)
    farTopRight = farCenter + camUp * (farHeight * 0.5) + camRight * (farWidth * 0.5)
    farBottomLeft = farCenter - camUp * (farHeight * 0.5) - camRight * (farWidth * 0.5)
    farBottomRight = farCenter - camUp * (farHeight * 0.5) + camRight * (farWidth * 0.5)

    nearTopLeft = nearCenter + camUp * (nearHeight * 0.5) - camRight * (nearWidth * 0.5)
    nearTopRight = (
        nearCenter + camUp * (nearHeight * 0.5) + camRight * (nearWidth * 0.5)
    )
    nearBottomLeft = (
        nearCenter - camUp * (nearHeight * 0.5) - camRight * (nearWidth * 0.5)
    )
    nearBottomRight = (
        nearCenter - camUp * (nearHeight * 0.5) + camRight * (nearWidth * 0.5)
    )

    if item_ids is None:
        item_ids = [-1] * 16

    lw = 2
    # Near pyramid
    item_ids[0] = p.addUserDebugLine(
        pos,
        nearTopLeft,
        [0, 1, 0],
        lw,
        duration,
        replaceItemUniqueId=item_ids[0],
        physicsClientId=cid,
    )
    item_ids[1] = p.addUserDebugLine(
        pos,
        nearTopRight,
        [0, 1, 0],
        lw,
        duration,
        replaceItemUniqueId=item_ids[1],
        physicsClientId=cid,
    )
    item_ids[2] = p.addUserDebugLine(
        pos,
        nearBottomLeft,
        [0, 1, 0],
        lw,
        duration,
        replaceItemUniqueId=item_ids[2],
        physicsClientId=cid,
    )
    item_ids[3] = p.addUserDebugLine(
        pos,
        nearBottomRight,
        [0, 1, 0],
        lw,
        duration,
        replaceItemUniqueId=item_ids[3],
        physicsClientId=cid,
    )

    item_ids[4] = p.addUserDebugLine(
        pos,
        farTopLeft,
        [1, 0, 0],
        lw,
        duration,
        replaceItemUniqueId=item_ids[4],
        physicsClientId=cid,
    )
    item_ids[5] = p.addUserDebugLine(
        pos,
        farTopRight,
        [1, 0, 0],
        lw,
        duration,
        replaceItemUniqueId=item_ids[5],
        physicsClientId=cid,
    )
    item_ids[6] = p.addUserDebugLine(
        pos,
        farBottomLeft,
        [1, 0, 0],
        lw,
        duration,
        replaceItemUniqueId=item_ids[6],
        physicsClientId=cid,
    )
    item_ids[7] = p.addUserDebugLine(
        pos,
        farBottomRight,
        [1, 0, 0],
        lw,
        duration,
        replaceItemUniqueId=item_ids[7],
        physicsClientId=cid,
    )

    item_ids[8] = p.addUserDebugLine(
        nearTopLeft,
        nearTopRight,
        [0, 1, 0],
        lw,
        duration,
        replaceItemUniqueId=item_ids[8],
        physicsClientId=cid,
    )

    item_ids[9] = p.addUserDebugLine(
        nearTopRight,
        nearBottomRight,
        [0, 1, 0],
        lw,
        duration,
        replaceItemUniqueId=item_ids[9],
        physicsClientId=cid,
    )

    item_ids[10] = p.addUserDebugLine(
        nearBottomRight,
        nearBottomLeft,
        [0, 1, 0],
        lw,
        duration,
        replaceItemUniqueId=item_ids[10],
        physicsClientId=cid,
    )

    item_ids[11] = p.addUserDebugLine(
        nearBottomLeft,
        nearTopLeft,
        [0, 1, 0],
        lw,
        duration,
        replaceItemUniqueId=item_ids[11],
        physicsClientId=cid,
    )

    item_ids[12] = p.addUserDebugLine(
        farTopLeft,
        farTopRight,
        [0, 0, 1],
        lw,
        duration,
        replaceItemUniqueId=item_ids[12],
        physicsClientId=cid,
    )

    item_ids[13] = p.addUserDebugLine(
        farTopRight,
        farBottomRight,
        [0, 0, 1],
        lw,
        duration,
        replaceItemUniqueId=item_ids[13],
        physicsClientId=cid,
    )

    item_ids[14] = p.addUserDebugLine(
        farBottomRight,
        farBottomLeft,
        [0, 0, 1],
        lw,
        duration,
        replaceItemUniqueId=item_ids[14],
        physicsClientId=cid,
    )

    item_ids[15] = p.addUserDebugLine(
        farBottomLeft,
        farTopLeft,
        [0, 0, 1],
        lw,
        duration,
        replaceItemUniqueId=item_ids[15],
        physicsClientId=cid,
    )
    return item_ids


def draw_aabb(aabb, duration=0, phys_id=0):
    aabbMin = aabb[0]
    aabbMax = aabb[1]
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 0, 0], lifeTime=duration, physicsClientId=phys_id)
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [0, 1, 0], lifeTime=duration, physicsClientId=phys_id)
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [0, 0, 1], lifeTime=duration, physicsClientId=phys_id)

    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], lifeTime=duration, physicsClientId=phys_id)

    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], lifeTime=duration, physicsClientId=phys_id)

    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], lifeTime=duration, physicsClientId=phys_id)

    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], lifeTime=duration, physicsClientId=phys_id)

    f = [aabbMax[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], lifeTime=duration, physicsClientId=phys_id)

    f = [aabbMin[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], lifeTime=duration, physicsClientId=phys_id)

    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(
        f, t, [1.0, 0.5, 0.5], lifeTime=duration, physicsClientId=phys_id
    )
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], lifeTime=duration, physicsClientId=phys_id)
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], lifeTime=duration, physicsClientId=phys_id)


drawAABB = draw_aabb


def draw_cloud(
    pcd: "open3d.geometry.PointCloud",
    subsample_size=5000,
    pcd_debug_item: int = -1,
    phys_id: int = 0,
):
    all_points = np.asarray(pcd.points)
    ids = np.random.choice(np.arange(0, all_points.shape[0]), subsample_size)

    colors = np.zeros_like(pcd.colors)[ids, :]

    points = np.asarray(pcd.points)[ids, :]

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=phys_id)

    if pcd_debug_item >= 0:
        p.addUserDebugPoints(
            points,
            colors,
            lifeTime=0,
            pointSize=2,
            replaceItemUniqueId=pcd_debug_item,
            physicsClientId=phys_id,
        )
    else:
        pcd_debug_item = p.addUserDebugPoints(points, colors, lifeTime=0, pointSize=2)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    return pcd_debug_item
