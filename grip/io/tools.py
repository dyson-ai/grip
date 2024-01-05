import time
import warnings
import pickle
from os.path import exists, join, dirname, abspath
import os
import contextlib
import importlib

from .logging_utils import log

from functools import wraps

PROF_DATA = {}


def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling


def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        print("Function %s called %d times. " % (fname, data[0]))
        print("Execution time max: %.3f, average: %.3f" % (max_time, avg_time))


def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}


def get_package_path(package_name="grip_assets", path_suffix=""):
    m = None
    try:
        m = importlib.import_module(package_name)
    except ModuleNotFoundError as exec:
        log.info("Module {} was not found.".format(package_name))
        if package_name == "grip_data":
            log.info(
                "The package grip_data can be downloaded, would you like to do that now?"
            )
            answer = input("(y/n)? ")
            if answer in ["y", "Y"]:
                gs = importlib.import_module("grip_assets")
                gs.fetch_data()

                log.info("Package grip_data was downloaded and installed.")

                m = importlib.import_module(package_name)
        else:
            raise ModuleNotFoundError from exec

    package_path = dirname(m.__file__)

    package_path = join(package_path, path_suffix)

    return package_path


def get_abs_path(path):
    return abspath(path)


def crawl(path, extension_filter="urdf"):
    crawled_paths = dict()

    repeat_count = 0
    for dirpath, _, files in os.walk(path):
        for file in files:
            prefix = ""
            key = file
            if crawled_paths.get(file, None) is not None:
                prefix = "{:04d}".format(repeat_count)
                key = "{}_{}".format(prefix, file)
                repeat_count += 1

            if extension_filter is None:
                crawled_paths[key] = os.path.join(dirpath, file)
            elif file.split(".")[-1] == extension_filter:
                crawled_paths[key] = os.path.join(dirpath, file)

    return crawled_paths


def file_list(path, extension_filter="urdf"):
    files = crawl(path, extension_filter=extension_filter)

    fs = [file_path for _, file_path in files.items()]
    fs.sort()

    return fs


def get_file_path(file, search_paths):
    if not isinstance(search_paths, list):
        search_paths = [search_paths]

    crawled_paths = {}
    ext = file.split(".")[-1]

    for path in search_paths:
        crawled_paths.update(crawl(path, ext))

    file_path = crawled_paths.get(file, None)

    if file_path is None:
        log.warning("File %s not found, returning None path." % (file,))

    return file_path


def save_data(data, filename, over_write_existing=False):
    if not over_write_existing:
        if exists(filename):
            warnings.warn("File exist by same name, renaming new file...")
            filename = (
                filename[:-4]
                + time.strftime("_%b_%d_%Y_%H_%M_%S", time.localtime())
                + ".pkl"
            )

    output = open(filename, "wb")

    # Pickle dictionary using protocol 0.
    pickle.dump(data, output)

    # Pickle the list using the highest protocol available.
    # pickle.dump(selfref_list, output, -1)

    output.close()


def load_data(filename):
    try:
        pkl_file = open(filename, "rb")
    except OSError as e:
        raise e

    data = pickle.load(pkl_file)

    pkl_file.close()

    return data


def import_module(module_name):
    lib = None
    try:
        lib = importlib.import_module(module_name)
    except:
        log.debug(f"Failed to import module: {module_name}. Has it been installed?")

    return lib


@contextlib.contextmanager
def pushd(new_dir):
    old_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(old_dir)


def read_file_as_str(filename):
    with open(filename, "r") as file:
        data_str = file.read().rstrip()

    return data_str
