from typing import Tuple, List, Optional
import numpy as np
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.duration import Duration


class TFPublisher:
    """A transform publisher utility class. It can be used to broadcast transform to ROS TF tree."""

    def __init__(self, parent_node: "rclpy.node.Node"):
        """Contruscts a TFPublisher object

        Args:
            parent_node (rclpy.node.Node): parent node to which this TFPublisher is associated to.
        """

        self.node = parent_node

        self.tf_broadcaster = TransformBroadcaster(self.node)

    def broadcast_frame(
        self,
        transform: Tuple[List[float], List[float]],
        source_frame: Optional[str] = "base_link",
        target_frame: Optional[str] = "map",
        current_time: Optional["rclpy.time.Time"] = None,
    ):
        """Broadcasts a transform from source to target

        Args:
            transform (Tuple[List[float], List[float]]): a transform consisting of a position and quaternion tuple ([x,y,z], [qx, qy, qz, qw])
            source_frame (Optional[str], optional): the source frame. Defaults to "base_link".
            target_frame (Optional[str], optional): the target frame. Defaults to "map".
            current_time (Optional[&quot;rclpy.time.Time&quot;], optional): current time. Defaults to None.
        """
        t = TransformStamped()

        t.header.stamp = (
            self.node.get_clock().now().to_msg()
            if current_time is None
            else current_time
        )

        translation = transform[0]
        quaternion = transform[1]

        t.header.frame_id = source_frame
        t.child_frame_id = target_frame

        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]

        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        self.tf_broadcaster.sendTransform(t)


class TFListener:
    """A transform listener utility class. It can be used query transformations from ROS TF tree."""

    def __init__(self, parent_node: "rclpy.node.Node", buffer_span: float = 0):
        """Constructor to TFListener

        Args:
            parent_node (rclpy.node.Node): parent node to which this TFListener is associated to.
            buffer_span (float, optional): buffer time span in seconds
        """

        self.node = parent_node

        self.tf_buffer = Buffer(Duration(seconds=buffer_span))
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

    def get_transform(
        self,
        base_frame: str,
        source_frame: str,
        time: Optional["rclpy.time.Time"] = None,
        n_trials: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, "rclpy.time.Time"]:
        """gets transform between base_frame and source_frame at specified time

        Args:
            base_frame (str): base frame name
            source_frame (str): source frame name
            time (Optional[&quot;rclpy.time.Time&quot;], optional):  The time at which to get the transform (0 will get the latest). Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, &quot;rclpy.time.Time&quot;]: a tuple representing (position, quaternion, time)
        """
        if base_frame == source_frame:
            return (0, 0, 0), (0, 0, 0, 1), self.node.get_clock()
        for i in range(n_trials):
            t = None
            try:
                t = self.tf_buffer.lookup_transform(base_frame, source_frame, time)

                break

            except Exception as e:
                print("Exception type: ", type(e))
                self.node.get_logger().warning(
                    f"Transform could not be queried. Attempt {i}. Warning: {e}"
                )

                if i == (n_trials - 1):
                    self.node.get_logger().error(
                        f"Transform could not be queried. Error: {e}"
                    )
                    return None, None, None

        translation = (
            t.transform.translation.x,
            t.transform.translation.y,
            t.transform.translation.z,
        )
        quaternion = (
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        )

        return (
            np.array(translation).astype(np.float32),
            np.array(quaternion).astype(np.float32),
            time,
        )
