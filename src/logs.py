import logging
import os

def setup_logger(
    name=None,
    level=logging.INFO,
    log_path="logs/app.log",
    console=None,
):
    logger = logging.getLogger(name if name else __name__)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    if console is None:
        console = os.environ.get("LOG_STDOUT", "1") != "0"
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(levelname)s %(name)s: %(message)s"
    )
    if console:
        handler = logging.StreamHandler()
        handler.setFormatter(console_formatter)
        logger.addHandler(handler)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    return logger

def main():
    pass

if __name__ == "__main__":
    main()
