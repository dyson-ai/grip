import abc


class Entity(abc.ABC):
    """
    Every entity has a state
    """

    @abc.abstractmethod
    def state(self):
        """
        Reads current entity state
        """
        raise NotImplementedError("state: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def reset(self):
        """
        Restarts this entity to its original state
        """
        raise NotImplementedError("reset: NO EFFECT, NOT IMPLEMENTED")
