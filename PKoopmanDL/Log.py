import logging

LOGGER = logging.getLogger("PKoopmanDL")
LOGGER.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)s: [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)


def set_level(level):
  LOGGER.setLevel(level)


def check_level(level):
  # if effective level is less than or equal to level, return True
  return LOGGER.getEffectiveLevel() <= level


def debug_level():
  return check_level(logging.DEBUG)


def debug_message(message):
  LOGGER.debug(message)


def info_message(message):
  LOGGER.info(message)
