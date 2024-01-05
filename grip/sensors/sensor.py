import abc


class Sensor(abc.ABC):
    @abc.abstractmethod
    def obs(self):
        """
        Retrieves an observation of this sensor
        """
        raise NotImplementedError("obs: NO EFFECT, NOT IMPLEMENTED")
