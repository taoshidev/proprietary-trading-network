import logging


class LoggerUtils:
    @staticmethod
    def init_logger(logger_name):
        # Create a logger
        logger = logging.getLogger(logger_name)

        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)
        return logger
