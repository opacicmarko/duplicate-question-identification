import logging

def log(logger, msg, log_level = logging.INFO, print_msg: bool = True):
    logger.log(log_level, msg)
    if print_msg:
        print(msg)