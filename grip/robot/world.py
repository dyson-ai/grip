from dataclasses import dataclass
import pybullet as p
from ..io import log
from .entity import Entity
import atexit
import sys, os, gc
from typing import List, Callable, Optional, Union
import functools, inspect
import pybullet_data
import pkgutil


class BulletClient:
    def __init__(self, phys_id, pid):
        self._pid = pid
        self._client = phys_id

    def __getattr__(self, name):
        attribute = getattr(p, name)
        if inspect.isbuiltin(attribute):
            attribute = functools.partial(attribute, physicsClientId=self._client)
        if name == "disconnect":
            self._client = -1
        return attribute


class BulletWorld:
    """
    The BulletWorld represents an interface and connection to a Bullet physics simulation.

    """

    def __init__(self, **kwargs):
        """
        An object of this class is reponsible for esblishing communication to a new or existing bullet physics engine.

        Args:
            phys_opt (str): type of connection to bullet, it can assume one of the following values ['direct', 'gui', 'shared_memory', 'udp', 'tcp'].
                "direct": physics engine world is created headless, without GUI.

                "gui": physics engine world is created with opengl rendering as GUI.

                "shared_memory": connects to an existing physics server via shared memory.

                "udp": connects to an already existing physics server with the UDP protocol
                    you must provide hostname and port if you choose this option.

                "tcp": physics engine world is created connecting to an already existing physics server with the TCP communication.
                    You must provide hostname and port if you choose this option.

            gravity (list[float], optional): gravity vector. Defaults to [0, 0, -9.81].
            real_time_sim (bool, optional): If you enable the real-time simulation, you don't need to call 'world.step()'.
                Has no effect if world phys_opt is connected as 'direct', in this case you will have to call world.step(). Defaults to True.
            deformable (bool, optional): configure world to support deformable physics. Defaults to False.
            solver_iterations (int, optional): internal parameter representing maximum number of constraint solver iterations.Defaults to 240.
            timestep (float, optional): time step used for physics evolution. Defaults to 1/240.
            reset_at_init (bool, optional): if this is true, the method reset() will be called as this object is created.

        """

        self.phys_opt = kwargs.get("phys_opt", "direct")
        self.hostname = kwargs.get("hostname", "localhost")
        self.port = kwargs.get("port", 1234)
        self.gravity = kwargs.get("gravity", [0, 0, -9.81])
        self.real_time_sim = kwargs.get("real_time_sim", True)
        self.deformable = kwargs.get("deformable", False)
        self.solver_iterations = kwargs.get("solver_iterations", 240)
        self.timestep = kwargs.get("timestep", 1.0 / 240.0)
        self.reset_at_init = kwargs.get("reset_at_init", True)
        self.texture_randomiser = None
        self.phys_id = (
            -1
        )  # invalid phys_id initialised by default (-1 means disconected)
        options = "--background_color_red=0.2 --background_color_blue=0.2 --background_color_green=0.2"

        flag = p.DIRECT
        if self.phys_opt == "shared_memory":
            flag = p.SHARED_MEMORY
        elif self.phys_opt in ["gui", "GUI"]:
            flag = p.GUI
        elif self.phys_opt in ["gui_server", "GUI_SERVER"]:
            flag = p.GUI_SERVER
        elif self.phys_opt in ["UDP", "udp"]:
            flag = p.UDP
        elif self.phys_opt in ["TCP", "tcp"]:
            flag = p.TCP
        elif self.phys_opt == "egl":
            flag = p.SHARED_MEMORY_SERVER

        if self.phys_opt in [
            "gui",
            "GUI",
            "gui_server",
            "direct",
            "DIRECT",
            "shared_memory",
            "egl",
        ]:
            self.phys_id = p.connect(
                flag, options=options
            )  # (pb.DIRECT), pb.SHARED_MEMORY, pb.GUI #"--opengl2"
            if self.reset_at_init:
                self.reset()

        elif self.phys_opt in ["UDP", "udp"]:
            self.phys_id = p.connect(flag, self.hostname, self.port)
            p.setTimeOut(10)
        elif self.phys_opt in ["TCP", "tcp"]:
            log.info("Trying to connect to physics")
            self.phys_id = p.connect(flag, self.hostname, self.port)
            p.setTimeOut(10)

        if self.phys_id < 0:
            log.info("Failed to connect to physics engine")

        self.state_id = None

        self._pid = os.getpid()

        self.client = BulletClient(self.phys_id, self._pid)

        atexit.register(self.on_shutdown)

        self.entity_list = []
        self.keyboard_handlers = {ord("q"): self.shutdown}

        if self.phys_opt != "shared_memory":
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.image_textures = {}

        self._shutdown_callbacks = []

        if self.phys_opt == "egl":
            loaded_egl = self.load_egl_plugin()

            if loaded_egl:
                log.info("EGL plugin loaded successfully!")
            else:
                log.warning("EGL plugin failed to load")

    def load_egl_plugin(self) -> bool:
        """Loads EGL plugin

        Returns:
            (bool): whether or not the plugin was loaded successfully
        """

        egl = pkgutil.get_loader("eglRenderer")
        if egl:
            self.plugin_id = p.loadPlugin(
                egl.get_filename(), "_eglRendererPlugin", physicsClientId=self.phys_id
            )
        else:
            self.plugin_id = p.loadPlugin(
                "eglRendererPlugin", physicsClientId=self.phys_id
            )
        log.info(f"pluginId={self.plugin_id}")

        return self.plugin_id > 0

    def unload_egl_plugin(self) -> None:
        """Unloads EGL plugin"""

        if self.plugin_id > 0:
            p.unloadPlugin(self.plugin_id, physicsClientId=self.phys_id)

    def __del__(self):
        """Clean up connection if not already done."""

        # log.info("Clearing world resources! Current phys_id: {} pid: {}, os.getpid(): {}".format(self.id, self._pid, os.getpid()))
        self.disconnect()

    def disconnect(self) -> None:
        gc.collect()
        if self.id >= 0 and self._pid == os.getpid():
            try:
                # log.info("Attempting disconnecting this engine")

                p.removeAllUserParameters(physicsClientId=self.id)
                p.removeAllUserDebugItems(physicsClientId=self.id)

                self.clear_entities()

                p.disconnect(physicsClientId=self.id)
                self.phys_id = -1
                # log.info("Disconnecting this engine was successful")
            except p.error:
                pass
                # log.info("Failed to clean: {}".format(e))

    @property
    def id(self):
        """
        int: The unique identifier of this physics simulation connection.
        """

        return self.phys_id

    def step(self) -> None:
        """
        Evolves this simulation by one step step.
        """

        if self.state_id is None:
            self.save_state()

        p.stepSimulation(physicsClientId=self.id)

        self.handle_keyboard()

    def reset(self, load_plane: bool = False) -> None:
        """
        Completely resets this world. Starts a fresh world without any entities.
        """

        if self.deformable:
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD, physicsClientId=self.id)
        else:
            p.resetSimulation(physicsClientId=self.id)

        if self.real_time_sim:
            p.setRealTimeSimulation(1, physicsClientId=self.id)
        else:
            p.setTimeStep(self.timestep, physicsClientId=self.id)

        p.setPhysicsEngineParameter(
            numSolverIterations=self.solver_iterations, physicsClientId=self.id
        )

        p.setGravity(*self.gravity, physicsClientId=self.id)

        if load_plane:
            p.loadURDF(
                os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0]
            )

    def set_real_time(self, real_time_sim: bool) -> None:
        """
        Sets real-time sim flag

        real_time_sim: whether or not to step the simulation according to its real-time-clock (RTC) on a separate thread.
        """

        self.real_time_sim = real_time_sim

        p.setRealTimeSimulation(self.real_time_sim, physicsClientId=self.id)

    def set_gravity(self, gravity: List[float]) -> None:
        """
        Sets gravity of the world

        Args:
            gravity (list[float]): three dimensional gravity vector.

        """

        self.gravity = gravity

        p.setGravity(*self.gravity, physicsClientId=self.id)

    def remove_debug_item(self, item_id: int) -> None:
        """
        Removes a debug item from the simulation identified by its item_id.

        Args:
            item_id (int): item unique ID to be removed.

        """

        p.removeUserDebugItem(item_id, physicsClientId=self.id)

    def shutdown(self) -> None:
        """
        Exists and closes this process. Internally, it calls sys.exit().

        """

        sys.exit()

    def on_shutdown(self) -> None:
        """
        Removes objects, GUI elements, entities and disconnects from the physics server

        """

        if self.is_connected():
            self.disconnect()

            # log.info("Exiting GRIP. Bye!")

        for cb in self._shutdown_callbacks:
            cb()

    def shutdown_register(self, shutdown_callback: Callable[[], None]):
        assert callable(shutdown_callback)

        self._shutdown_callbacks.append(shutdown_callback)

    def setDebugCamera(
        self, view_dist=1, view_yaw=0, view_pitch=1.0, view_target_pos=None, **kwargs
    ) -> None:
        """
        Sets GUI view camera, the observer view in the GUI.

        Args:
            view_dist (float): distance from eye to camera target position
            view_yaw (float): camera yaw angle (in degrees) left/right
            view_pitch (float): camera pitch angle (in degrees) up/down
            view_target_pos (list[float]): cameraTargetPosition is the camera focus point in 3D space.

        """
        if view_target_pos is None:
            view_target_pos = [0, 0, 0.5]
        p.resetDebugVisualizerCamera(
            view_dist, view_yaw, view_pitch, view_target_pos, physicsClientId=self.id
        )

    def hide_gui(self) -> None:
        """
        Hides GUI panels from debug visualiser
        """
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)

    def save_state(self) -> None:
        """
        Takes a snapshot and saves the current state of the simulation in-memory.
        This way the simulation can be efficiently restarted by calling restore_state()
        """

        if self.state_id is not None:
            p.removeState(self.state_id, physicsClientId=self.id)

        self.state_id = p.saveState(physicsClientId=self.id)

    def restore_state(self) -> None:
        """
        Restores the simulation to a previous state aved by the most recent call to save_state()
        """

        if self.state_id is not None:
            p.restoreState(self.state_id, physicsClientId=self.id)

    def is_connected(self) -> bool:
        """
        Tells if this world is still connected to the physics server

        Returns:
            bool: connection_status
        """

        return p.isConnected(physicsClientId=self.id)

    def set_data(self, entity: Entity, key: str, value: str) -> None:
        """
        Attaches a key-value pair to the provided entity. Can be used for storing a variable globally
        across shared memory instances of pybullet.

        Args:
            entity (Entity): the object to which the data should be attached
            key (str): the key to be used for accessing the object
            value (str): the value to be stored for the key
        """
        p.addUserData(entity.id, key, value, physicsClientId=self.id)

    def add_data(self, key: str, value: str):
        """
        Attaches a key-value pair to a new dummy object. Can be used for storing a variable globally
        across shared memory instances of pybullet. Creates a dummy object and attaches value using
        :py:meth:`set_data`.

        Args:
            key (str): the key to be used for accessing the object
            value (str): the value to be stored for the key
        """
        shape_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.01] * 3,
            rgbaColor=[0, 0, 0, 0],
            physicsClientId=self.id,
        )
        box_id = p.createMultiBody(
            0, -1, shape_id, [0] * 3, [0, 0, 0, 1], physicsClientId=self.id
        )

        @dataclass
        class dummy:
            id: int = box_id

        self.register_entity(dummy)
        self.set_data(dummy, "package_name", "panda_flex_dev")

    def get_data(self, query_key: str) -> Optional[str]:
        """
        Retrieves data from shared memory using provided key. Assumes key is unique across all enitities.
        (See :py:meth:`add_data` and :py:meth:`set_data`)

        Args:
            key (str): the key used for storing the object

        Returns:
            value (str): the value stored for the key. Returns None on failure.
        """
        bodies_uids = self.get_bodies_uids()
        for buid in bodies_uids:
            for i in range(p.getNumUserData(buid, physicsClientId=self.id)):
                (
                    userDataId,
                    key,
                    bodyId,
                    linkIndex,
                    visualShapeIndex,
                ) = p.getUserDataInfo(buid, i, physicsClientId=self.id)
                key = key.decode("utf-8")
                if key == query_key:
                    return p.getUserData(userDataId, physicsClientId=self.id).decode(
                        "utf-8"
                    )

        return None

    def get_bodies_uids(self) -> List[int]:
        """
        Get uids of all objects in the sim world.

        Returns:
            bodies_uids (List[int]): uids of all objects in the simulation.
        """
        n_bodies = p.getNumBodies(physicsClientId=self.id)

        bodies_uids = []
        for i in range(n_bodies):
            buid = p.getBodyUniqueId(i, physicsClientId=self.id)
            bodies_uids.append(buid)

        return bodies_uids

    def get_body_by_name(self, name: str) -> int:
        """
        Retrieves a pre-existing body unique identifier from the simulation searched by its name
        The name is typically given in the objects' URDF definition.

        Args:
            name (str): Name of the body as defined in its URDF.

        Returns:
            body_idx (int): unique identifier of the body in this world
        """

        n_bodies = p.getNumBodies(physicsClientId=self.id)

        body_idx = -1
        for i in range(n_bodies):
            buid = p.getBodyUniqueId(i, physicsClientId=self.id)
            info = p.getBodyInfo(buid, physicsClientId=self.id)
            log.info("BodyInfo {}: {}".format(i, info))

            body_name = info[1].decode("utf-8")

            if body_name == name:
                body_idx = buid
                break

        if body_idx < 0:
            log.debug("Body name {} not found")

        return body_idx

    def get_body_names(self) -> List[str]:
        """
        Retrieves all pre-existing bodies names that are given in the objects' URDF definition.

        Returns:
            body_names (list[str]): list of names for each  in this world
        """

        n_bodies = p.getNumBodies(physicsClientId=self.id)

        bodies = []
        for i in range(n_bodies):
            buid = p.getBodyUniqueId(i, physicsClientId=self.id)
            info = p.getBodyInfo(buid, physicsClientId=self.id)
            log.info("BodyInfo {}: {}".format(buid, info))

            body_name = info[1].decode("utf-8")

            bodies.append(body_name)

        return bodies

    def register_keyhandler(self, key_code: int, func: Callable[[], None]) -> None:
        """
        Registers a callback function for a given key_code

        Args:
            key_code (int): int
                key code, can be obtained using builtin function ord(..)
            func (callable):
                a callback function that does not receive any parameters.
                the signature of the callback function should be `def fun() -> None`

        Returns:
            body_idx (int): unique identifier of the body in this world

        Examples:
            >>> world.register_keyhandler(ord('a'), callback_fun)
        """

        self.keyboard_handlers[key_code] = func

    def register_entity(self, entity: Union[Entity, List[Entity]]) -> None:
        """
        Registers an entity or list of entities to this world for memory management purposes
        This allows entities registered to this world to be removed on shutdown

        This is useful if this world is connected to a central physics server,
        which in turn the physics server may be connected to many other worlds.
        When each world adds its entities, each world then becomes responsible for removing them
        after they are shutdown.


        Args:

            entity (Union[Entity,List[Entity]]): entity or entities to be registed, individual entities must inherit from grip.robot.Entity
                and implement the basic grip.robot.Entity interface.

        """

        if isinstance(entity, list):
            self.entity_list.extend(entity)
        else:
            self.entity_list.append(entity)

        p.configureDebugVisualizer(
            p.COV_ENABLE_MOUSE_PICKING, 1, physicsClientId=self.id
        )

    def handle_keyboard(self) -> None:
        """
        Handles keyboard events that were register with register_keyhandler()

        """

        keys = p.getKeyboardEvents(physicsClientId=self.id)

        reg_keys = self.keyboard_handlers.keys()

        for key in keys:
            if keys[key] & p.KEY_WAS_TRIGGERED:
                log.info("Key {} was pressed.".format(key))

        for key in reg_keys:
            if key in keys and keys[key] & p.KEY_WAS_TRIGGERED:
                self.keyboard_handlers[key]()

    def clear_entities(self) -> None:
        """
        Clear all entities registered to this world

        """

        for entity in self.entity_list:
            p.removeBody(entity.id, physicsClientId=self.id)
        self.entity_list = []

    def remove_entity(self, entity_id: int) -> bool:
        """
        Clear a specific entity provided its unique identifier

        Args:
        entity_id (int): unique identifier of the entity

        Returns:
            bool: whether or not the removal was successfull.
        """

        success = False
        try:
            p.removeBody(entity_id, physicsClientId=self.id)
            success = True
            if entity_id in self.entity_list:
                self.entity_list.remove(entity_id)

        except Exception as e:
            log.warning(
                f"Problem when trying to remove entity id: {entity_id}. Message {e}"
            )
        finally:
            return success

    def texture(self, texture_path: str) -> int:
        """
        Retrieves a texture's unique identifier (uid) given its path.
        If a texture was already loaded in memory with the same path, then the existing texture id will be retrived.
        Otherwise, the texture will be loaded and its uid will be returned.


        Args:
            texture_path (str): full path of texture file (typically a .png file)
        Returns:
            int: unique identifier of loaded texture
        """

        if self.image_textures.get(texture_path, None) is None:
            self.add_texture(texture_path)

            log.debug("Loading to memory!!")
        else:
            log.debug(f"Hit!! {texture_path}")

        return self.image_textures[texture_path]

    def add_texture(self, texture_path: str) -> None:
        """
        Loads texture located in texture_path to memory and adds to this world's texture dictionary.


        Args:
            texture_path (str): full path of texture file (typically a .png file)
        """

        self.image_textures[texture_path] = p.loadTexture(
            texture_path, physicsClientId=self.id
        )

        log.info(f"Loaded texture {texture_path}: {self.image_textures[texture_path]}")
