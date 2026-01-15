import os

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data folder
_PACKAGE_ROOT = os.path.join(_PROJECT_ROOT, "src/call_of_birds_autobird")  # root of source folder
_DATACLASSES_ROOT = os.path.join(_PACKAGE_ROOT, "call_of_func/dataclasses")  # root of dataclasses folder
_PATH_PREPROCESSING = os.path.join(_DATACLASSES_ROOT, "Preprocessing.py")  # Preprocessing.py path
