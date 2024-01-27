import datetime
import logging
import sys


logger_initialized = False


def create_default_formatter() -> logging.Formatter:
    formatter = logging.Formatter('haplo [{asctime} {levelname} {name}] {message}', style='{')
    return formatter


def set_up_default_logger():
    global logger_initialized
    if not logger_initialized:
        formatter = create_default_formatter()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger = logging.getLogger('haplo')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        sys.excepthook = excepthook
        logger_initialized = True


def excepthook(exc_type, exc_value, exc_traceback):
    logger = logging.getLogger('haplo')
    logger.critical(f'Uncaught exception at {datetime.datetime.now()}:')
    logger.handlers[0].flush()
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
