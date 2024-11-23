import logging

# set logging format: [time][level][module][line]: message
# module should indicate the path to the file
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(module)s][%(lineno)d]: %(message)s",
    level=logging.INFO,
)
__version__ = "0.0.1"
