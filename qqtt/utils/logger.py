"""Simple colored logger used throughout the repository."""

from __future__ import annotations

import logging
import os.path
import time
from typing import Optional

from .misc import singleton, master_only
from termcolor import colored
import sys


class Formatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    time_str = "%(asctime)s"
    level_str = "[%(levelname)7s]"
    msg_str = "%(message)s"
    file_str = "(%(filename)s:%(lineno)d)"

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class SteamFormatter(Formatter):

    FORMATS = {
        logging.DEBUG: colored(Formatter.msg_str, "cyan"),
        logging.INFO: colored(
            " ".join([Formatter.time_str, Formatter.level_str, ""]),
            "white",
            attrs=["dark"],
        )
        + colored(Formatter.msg_str, "white"),
        logging.WARNING: colored(
            " ".join([Formatter.time_str, Formatter.level_str, ""]),
            "yellow",
            attrs=["dark"],
        )
        + colored(Formatter.msg_str, "yellow"),
        logging.ERROR: colored(
            " ".join([Formatter.time_str, Formatter.level_str, ""]),
            "red",
            attrs=["dark"],
        )
        + colored(Formatter.msg_str, "red")
        + colored(" " + Formatter.file_str, "red", attrs=["dark"]),
        logging.CRITICAL: colored(
            " ".join([Formatter.time_str, Formatter.level_str, ""]),
            "red",
            attrs=["dark", "bold"],
        )
        + colored(
            Formatter.msg_str,
            "red",
            attrs=["bold"],
        )
        + colored(" " + Formatter.file_str, "red", attrs=["dark", "bold"]),
    }


class FileFormatter(Formatter):

    FORMATS = {
        logging.INFO: " ".join(
            [Formatter.time_str, Formatter.level_str, Formatter.msg_str]
        ),
        logging.WARNING: " ".join(
            [Formatter.time_str, Formatter.level_str, Formatter.msg_str]
        ),
        logging.ERROR: " ".join(
            [
                Formatter.time_str,
                Formatter.level_str,
                Formatter.msg_str,
                Formatter.file_str,
            ]
        ),
        logging.CRITICAL: " ".join(
            [
                Formatter.time_str,
                Formatter.level_str,
                Formatter.msg_str,
                Formatter.file_str,
            ]
        ),
    }


@singleton
class ExpLogger(logging.Logger):

    def __init__(self, name: Optional[str] = None) -> None:
        if name is None:
            name = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(time.time()))
        super().__init__(name)
        self.setLevel(logging.DEBUG)

        self.set_log_stream()
        self.filehandler = None

    @master_only
    def set_log_stream(self) -> None:
        self.stearmhandler = logging.StreamHandler()
        self.stearmhandler.setFormatter(SteamFormatter())
        self.stearmhandler.setLevel(logging.DEBUG)

        self.addHandler(self.stearmhandler)

    def remove_log_stream(self) -> None:
        self.removeHandler(self.stearmhandler)

    @master_only
    def set_log_file(self, path: str, name: Optional[str] = None) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(
            path, f"{self.name}.log" if name is None else f"{name}.log"
        )
        self.filehandler = logging.FileHandler(file_path)
        self.filehandler.setFormatter(FileFormatter())
        self.filehandler.setLevel(logging.INFO)
        self.addHandler(self.filehandler)

    @master_only
    def info(self, msg: str, **kwargs) -> None:
        return super().info(msg, **kwargs)

    @master_only
    def warning(self, msg: str, **kwargs) -> None:
        return super().warning(msg, **kwargs)

    @master_only
    def error(self, msg: str, **kwargs) -> None:
        return super().error(msg, **kwargs)

    @master_only
    def debug(self, msg: str, **kwargs) -> None:
        return super().debug(msg, **kwargs)

    @master_only
    def critical(self, msg: str, **kwargs) -> None:
        return super().critical(msg, **kwargs)


logger = ExpLogger()

class StreamToLogger:
    """Redirect ``stdout``/``stderr`` streams to a :class:`logging.Logger`."""

    def __init__(self, logger: logging.Logger, log_level: int) -> None:
        super().__init__()
        self.logger = logger
        self.log_level = log_level

    def write(self, message: str) -> None:
        if message.strip():
            self.logger.log(self.log_level, message.strip())

    def flush(self) -> None:  # pragma: no cover - used by stream interface
        pass