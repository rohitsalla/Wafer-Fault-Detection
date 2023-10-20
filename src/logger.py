"""Logger.

This module handles all the logging mechanism.
"""
import logging


class AppLogger:
    """Logger class."""

    def __init__(self) -> None:
        """Initialize the required variables."""
        self.__log_format = (
            "%(asctime)s - [%(levelname)s] - [%(filename)s]"
            "- [%(lineno)s] - %(message)s"
        )

    def get_file_handler(self, log_file_name):
        """Write log to a file.

        Args:
            log_file_name (str): The name of the logfile
        Returns:
            FileHandler: FileHandler for logging to file
        """
        file_handler = logging.FileHandler(log_file_name, mode="a+")
        file_handler.setFormatter(logging.Formatter(self.__log_format))
        return file_handler

    def get_stream_handler(self):
        """Write log to console.

        Returns:
            StreamHandler: StreamHandler for logging to console
        """
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(self.__log_format))
        return stream_handler

    def get_logger(self, name, level=logging.INFO):
        """Get the logger with both file and console log on different levels.

        Args:
            name (str): The name of the logger
            level (int): log level
        Returns:
            logging: A logging object to use when logging in the code
        """
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            # Logger is already configured, remove all handlers
            logger.handlers = []
        logger.addHandler(self.get_file_handler(name))
        logger.addHandler(self.get_stream_handler())
        logger.setLevel(level)
        return logger
