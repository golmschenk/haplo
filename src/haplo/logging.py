import logging
import sys


def create_default_formatter() -> logging.Formatter:
    formatter = logging.Formatter('haplo [{asctime} {levelname} {name}] {message}', style='{')
    return formatter


def set_up_default_logger():
    formatter = create_default_formatter()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger = logging.getLogger('haplo')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


