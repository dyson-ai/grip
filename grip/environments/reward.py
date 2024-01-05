import abc


class Reward:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reward(self, env, *args, **kwargs):
        """
        Computes reward based on arbitrary list of parameters (application/task dependent)
        """
        raise NotImplementedError
