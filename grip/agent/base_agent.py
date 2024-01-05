import abc


class Agent(abc.ABC):
    @abc.abstractmethod
    def observe(self):
        """
        extract raw observation form state
        including postprocess to usable state if necessary
        """
        raise NotImplementedError

    @abc.abstractmethod
    def act(self):
        """
        Acts on the environment
        """
        raise NotImplementedError


class GraspAgent(Agent):
    @abc.abstractmethod
    def postprocess(self, obs):
        """
        postprocessing raw observation to agent input
        """
        raise NotImplementedError

    @abc.abstractmethod
    def call_grasp_generator(self, target_pcd, context_info):
        """
        the method to call corresponding generator service/topic with
        observation point cloud
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parse_grasp_config(self, grasp_config, auxilary_info):
        """
        formatting single solution provided by grasp generator to
        environment pos, quat to execute

        output can be single pose or pregrasp pose + grasping pose
        depend on whether the generator has already provided the
        pregrasp pose
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_grasp(self, solutions):
        """
        selectting solution with a cartesian planner with obstacles
        to see if both pre-grasp pose and grasping pose are feasible

        [home --> pregrasping pose]: collision free planning
        [pregrasping pose --> grasping pose]: collision ignored
        """
        raise NotImplementedError

    @abc.abstractmethod
    def grasp(self):
        """
        execute grasping according to postprocessed grasping solution
        """
        raise NotImplementedError
