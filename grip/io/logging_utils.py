import logging

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "not_set": logging.NOTSET,
}

log = None


class Logger:
    def __init__(self, name, level_name="debug"):
        self.configure(name, level_name)

    def configure(self, name, level_name="debug"):
        format_str = "[%(asctime)s][%(levelname)s]: %(message)s"

        level = LEVELS.get(level_name, logging.NOTSET)
        logging.basicConfig(format=format_str, level=level)

        self._logger = logging.getLogger(name)
        self._logger.propagate = False
        # otherwise handler additions are propagated to root logger resulting in double printing

        # creating console handler
        self._ch = logging.StreamHandler()
        self._ch.setLevel(level)

        self._formatter = logging.Formatter(format_str)
        self._ch.setFormatter(self._formatter)

        # add handler only if it doesn't already have one
        if not self._logger.handlers:
            self._logger.addHandler(self._ch)

        self._logger.setLevel(level)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)

    def setLevel(self, level_name):
        level = LEVELS.get(level_name, logging.NOTSET)

        self._ch.setLevel(level)
        self._logger.setLevel(level)


def logging_setup(name=None, level_name="debug"):
    global log
    log = Logger(name, level_name)
    return log


log = logging_setup("rosout", "info")


def set_log_level(level_name):
    level = LEVELS.get(level_name, logging.NOTSET)

    l = logging.getLogger("rosout")
    l.setLevel(level=level)
