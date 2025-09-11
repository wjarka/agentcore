import logging
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

current_logger = logging.getLogger(__name__)


def set_logger(logger: logging.Logger) -> None:
    global current_logger
    current_logger = logger


def logger() -> logging.Logger:
    return current_logger
