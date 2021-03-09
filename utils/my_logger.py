import logging
import sys, os



def my_logger(filename='my_log.log', out_level=10, err_level=40):

    """
    Logger out:
    1) print() / stdout -> to file.
    2) stderr -> to file and to console (by level)
    """

    ###
    original_stdout = sys.stdout # Save a reference to the original standard output
    sys.stdout = open(filename, 'w')
    ###

    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    console_handler = logging.StreamHandler(sys.stdout) #direct loggind to stdout
    console_handler.setLevel(out_level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')




    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)

    #sys.stdout = original_stdout # Reset the standard output to its original value

    """
    # 'application' code - > err message
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')
    """

    return logger


def create_log(log_name, dir_path, out_level=10, err_level=40):
    """Create custom logger
    (log_name, dir_path, log level, error (consol) level"""

    file_path = os.path.join(dir_path, log_name)
    return my_logger(file_path, out_level, err_level)



class StdoutLogger(object):

    """
    This will write all results being printed on stdout from
    the python source to file to the logfile.
    """

    def __init__(self, log_name):
        self.terminal = sys.stdout
        self.log = open(log_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass