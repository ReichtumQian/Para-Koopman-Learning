import logging

LOGGER = logging.getLogger("PKoopmanDL")
LOGGER.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)s: [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)


def debug(message):
  LOGGER.debug(message)


def info(message):
  LOGGER.info(message)


def warning(message):
  LOGGER.warning(message)


def error(message):
  LOGGER.error(message)
