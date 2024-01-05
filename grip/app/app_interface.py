import abc


class GripApp:
    @abc.abstractmethod
    def setup(self, **kwargs):
        """
        Not implemented
        """
        raise NotImplementedError("setup: NO EFFECT, NOT IMPLEMENTED")

    @abc.abstractmethod
    def run(self, **kwargs):
        """
        Not implemented
        """
        raise NotImplementedError("init: NO EFFECT, NOT IMPLEMENTED")
