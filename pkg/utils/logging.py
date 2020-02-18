# -*- coding:utf-8 -*-
import os
import os.path as osp
import time
from logging import (
    INFO,
    getLevelName,
    getLogger,
    Formatter,
    basicConfig,
    StreamHandler,
)
import sys

LOG_DEFAULT_FORMAT = "[ %(asctime)s][%(module)s.%(funcName)s] %(message)s"
LOG_DEFAULT_LEVEL = INFO


def strftime(t=None):
    """Get string of current time"""
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(t or time.time()))


def init_logging(logging_dir, filename=None, level=None, log_format=None):
    if not osp.exists(logging_dir):
        os.makedirs(logging_dir)

    filename = filename or strftime() + ".log"
    log_format = log_format or LOG_DEFAULT_FORMAT

    global LOG_DEFAULT_LEVEL
    if isinstance(level, str):
        level = getLevelName(level.upper())
    elif level is None:
        level = LOG_DEFAULT_LEVEL

    LOG_DEFAULT_LEVEL = level

    basicConfig(
        filename=osp.join(logging_dir, filename),
        format=log_format,
        level=level,
    )


def get_logger(name, level=None, log_format=None, print_to_std=True):
    """
    Get the logger

    level: if None, then use default=INFO
    log_format: if None, use default format
    print_to_std: default=True
    """
    if level is None:
        level = LOG_DEFAULT_LEVEL
    elif isinstance(level, str):
        level = getLevelName(level)

    if not log_format:
        log_format = LOG_DEFAULT_FORMAT
    logger = getLogger(name)
    logger.setLevel(level)

    if print_to_std:
        handler = StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(Formatter(log_format))
        logger.addHandler(handler)

    return logger
