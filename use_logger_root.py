from utils.my_logger import my_logger, create_log
import os


log_path = os.path.join(os.path.dirname(__file__), 'logs')
logger = create_log('root_log.log', log_path, 10, 20)

#
# logger = my_logger(file_path, 10, 20)
#
# logger.info(file_path)

def progression(start, stop, step):
    """Вывод последовательности функцкциональным прогр.
    Цикл заменяется рекурсией.
    """
    if start >= stop:
        return
    else:
        print(start)
        #logger.debug(start)
        progression(start + step, stop, step)

print("Progression:")
progression(0, 10, 1)


logger.debug('Quick zephyrs blow, vexing daft Jim.')
logger.info('How quickly daft jumping zebras vex.')
logger.warning('Jail zesty vixen who grabbed pay from quack.')
logger.error('The five boxing wizards jump quickly.')
