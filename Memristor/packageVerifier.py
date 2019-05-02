import importlib
from distutils.version import LooseVersion

# check that all packages are installed (see requirements.txt file)


def check_packages():
    message = ''
    required_packages = {'numpy',
                         'matplotlib',
                         'tensorflow',
                         'keras',
                         'PySimpleGUI',
                         'cv2',
                         'PIL'
                         }

    problem_packages = list()
    # Iterate over the required packages: If the package is not installed
    # ignore the exception.
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            if package == 'cv2':
                package = 'opencv-python'
            problem_packages.append(package)

    if len(problem_packages) is 0:
        message = 'All is well. All the required packages are installed. Enjoy!'
    else:
        message = 'The following packages are required but not installed: ' + \
            ', '.join(problem_packages)

    return message
