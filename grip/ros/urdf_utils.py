from pathlib import Path
from tempfile import NamedTemporaryFile

from ament_index_python.packages import get_package_share_path
from urdf_parser_py.urdf import URDF, Mesh


def normalise_urdf_paths(file: Path | str) -> str:
    """Normalise the robot description file to use absolute paths for the collision and visual tags.

    Args:
        file (pathlib.Path | str): a path to a URDF, or pre-loaded URDF string
    Returns:
        (str): a URDF file path with absolute paths to collision and visual tag mesh resources

    """
    if isinstance(file, str):
        robot = URDF.from_xml_string(file)
    elif isinstance(file, Path):
        robot = URDF.from_xml_file(file)

    for link in robot.links:
        for collision in link.collisions:
            if isinstance(
                collision.geometry,
                Mesh,
            ) and collision.geometry.filename.startswith("package://"):
                package_name, relative_path = collision.geometry.filename.split(
                    "package://",
                )[1].split("/", 1)
                collision.geometry.filename = (
                    f"file:/{get_package_share_path(package_name)}/{relative_path}"
                )
        for visual in link.visuals:
            if isinstance(
                visual.geometry,
                Mesh,
            ) and visual.geometry.filename.startswith("package://"):
                package_name, relative_path = visual.geometry.filename.split(
                    "package://",
                )[1].split("/", 1)
                visual.geometry.filename = (
                    f"file:/{get_package_share_path(package_name)}/{relative_path}"
                )

    with NamedTemporaryFile(
        mode="w",
        prefix="grip_",
        delete=False,
    ) as parsed_file:
        parsed_file_path = parsed_file.name
        parsed_file.write(robot.to_xml_string())
        return parsed_file_path
