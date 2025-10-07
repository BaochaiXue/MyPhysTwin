# Stage 0—Experiment Logger Utilities
# Role: Provide colourised console/file logging tailored to the inverse-physics stack;
#       ensure singleton logger so modules share handlers configured per run.
# Inputs: Log records emitted via `logger.info/debug/...`, optional path/name for
#         file logging, stderr/stdout redirection for stream capture.
# Outputs: Console output with ANSI colours, log files under `temp_experiments/<case>`.
# Key in-house deps: `.misc.singleton` to enforce single instance, `.misc.master_only`
#                    to gate logging in distributed runs.
# Side effects: Creates directories/files for logs, mutates global handlers, can
#               intercept `sys.stdout`/`sys.stderr` via `StreamToLogger`.
# Assumptions: Caller ensures `set_log_file` path exists or is creatable; singleton
#              instance shared process-wide.

import logging
import os.path
import time
from typing import Optional

from .misc import singleton, master_only
from termcolor import colored
import sys


class Formatter(logging.Formatter):
    """Base formatter supplying reusable format tokens for console/file handlers.

    Subclasses populate the `FORMATS` map with strings that leverage the shared tokens
    (`time_str`, `level_str`, etc.) so they render consistent log layouts across
    streams.
    """

    time_str = "%(asctime)s"
    level_str = "[%(levelname)7s]"
    msg_str = "%(message)s"
    file_str = "(%(filename)s:%(lineno)d)"

    def format(self, record):
        """Emit a formatted log line using the template tied to the record level.

        Parameters
        ----------
        record : logging.LogRecord
            Log record yielded by the Python `logging` system; `levelno` selects the
            template in `self.FORMATS`.

        Returns
        -------
        str
            Fully rendered log message ready for printing or file writing.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class SteamFormatter(Formatter):
    """Console formatter that applies ANSI colours based on severity levels."""

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
    """File formatter that omits colours while preserving timestamp/level fields."""

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
    """Singleton logger that wires coloured console output and per-run log files.

    Mirrors Python's `logging.Logger` API but ensures handlers are configured once per
    process and that logging only happens on the master rank when running distributed
    jobs.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialise the logger name and install the default stream handler.

        Parameters
        ----------
        name : Optional[str]
            Explicit logger name; when omitted we stamp the current timestamp so each
            run writes to a unique logfile.
        """

        if name is None:
            name = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(time.time()))
        super().__init__(name)
        self.setLevel(logging.DEBUG)

        self.set_log_stream()
        self.filehandler = None

    @master_only
    def set_log_stream(self):
        """Attach a colourised stream handler for stdout/stderr diagnostics."""
        self.stearmhandler = logging.StreamHandler()
        self.stearmhandler.setFormatter(SteamFormatter())
        self.stearmhandler.setLevel(logging.DEBUG)

        self.addHandler(self.stearmhandler)

    def remove_log_stream(self):
        """Detach the previously registered stream handler (useful for cleanup)."""
        self.removeHandler(self.stearmhandler)

    @master_only
    def set_log_file(self, path: str, name: Optional[str] = None):
        """Create a log file handler that mirrors INFO+ records to disk.

        Parameters
        ----------
        path : str
            Directory where the log file should be created; created if missing.
        name : Optional[str]
            Optional filename stem; defaults to the logger name when omitted.

        Side Effects
        ------------
        * Creates `path` on disk if it does not yet exist.
        * Replaces `self.filehandler` with the newly constructed file handler.
        """
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
    def info(self, msg, **kwargs) -> None:
        """Proxy `logging.Logger.info` but guard execution to master ranks."""
        return super().info(msg, **kwargs)

    @master_only
    def warning(self, msg, **kwargs) -> None:
        """Proxy `warning` while respecting master-only execution semantics."""
        return super().warning(msg, **kwargs)

    @master_only
    def error(self, msg, **kwargs) -> None:
        """Proxy `error` while respecting master-only execution semantics."""
        return super().error(msg, **kwargs)

    @master_only
    def debug(self, msg, **kwargs) -> None:
        """Proxy `debug` while respecting master-only execution semantics."""
        return super().debug(msg, **kwargs)

    @master_only
    def critical(self, msg, **kwargs) -> None:
        """Proxy `critical` while respecting master-only execution semantics."""
        return super().critical(msg, **kwargs)


logger = ExpLogger()

class StreamToLogger():
    """File-like shim that forwards writes to a logger handler.

    Useful when third-party code expects a stream (e.g., `sys.stdout`) but we want the
    messages to funnel through the structured logging infrastructure instead.
    """

    def __init__(self, logger, log_level):
        """Store the logger target and severity level.

        Parameters
        ----------
        logger : logging.Logger
            Destination logger instance receiving forwarded messages.
        log_level : int
            Logging level (e.g., `logging.INFO`) to tag forwarded records with.
        """
        super().__init__()
        self.logger = logger
        self.log_level = log_level

    def write(self, message):
        """Forward buffered text to the logger if it contains non-whitespace."""
        if message.strip():
            self.logger.log(self.log_level, message.strip())

    def flush(self):
        """No-op; included for file-like compatibility with standard streams."""
        pass
