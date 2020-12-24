import os
import shutil

def try_func(func):
    """
    Try / except wrapper
    """
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)  # execute function

        except NameError:
            print("NAME ERROR in {:s} - not executed".format(str(func)))
        except IOError:
            print("FILE ERROR in {:s} - not executed".format(str(func)))
    return wrapper

@try_func
def mk_run_folders(run_folder, name1, name2, name3):

    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
        os.mkdir(os.path.join(run_folder, str(name1)))
        os.mkdir(os.path.join(run_folder, str(name2)))
        os.mkdir(os.path.join(run_folder, str(name3)))
        print('new dirs OK:', run_folder, name1, name2, name3)
    else:
        print('run dirs OK')

@try_func
def copy_weights(run_folder, suff='.data-00000-of-00001', add='_'):

    suff = '.data-00000-of-00001'
    shutil.copy(os.path.join(run_folder, 'weights/weights'+ suff), os.path.join(run_folder, 'weights/weights' + add + suff))
    shutil.copy(os.path.join(run_folder, 'weights/weights.index'), os.path.join(run_folder, 'weights/weights_.index'))
    print('copy OK:', os.path.join(run_folder, 'weights/weights' + add + suff))


def timer(func):

    """Функция-декоратор измерения времени выполнения функции
    @decorator
    def my_function():
        # do smth
    """
    import datetime

    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()  # TIMER ON

        func(*args, **kwargs)  # execute function

        time_el = datetime.datetime.now() - start_time  #TIMER OFF
        print('Time elapsed: min {:.0f}, sec {:.0f}'.format(time_el.total_seconds() // 60, time_el.total_seconds() % 60))
    return wrapper


def benchmark(func):
    """ Точный таймер """
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('[*] Время выполнения: {} секунд.'.format(end-start))
    return wrapper
