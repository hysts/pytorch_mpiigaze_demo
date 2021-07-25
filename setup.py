import pathlib

from setuptools import find_packages, setup


def _get_long_description():
    path = pathlib.Path(__file__).parent / 'README.md'
    with open(path, encoding='utf-8') as f:
        long_description = f.read()
    return long_description


def _get_requirements(path):
    with open(path) as f:
        data = f.readlines()
    return data


setup(
    name='ptgaze',
    version='0.2.0',
    author='hysts',
    url='https://github.com/hysts/pytorch_mpiigaze_demo',
    python_requires='>=3.7',
    install_requires=_get_requirements('requirements.txt'),
    packages=find_packages(exclude=('tests', )),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'ptgaze=ptgaze.main:main',
        ],
    },
    description='Gaze estimation using MPIIGaze and MPIIFaceGaze',
    long_description=_get_long_description(),
    long_description_content_type='text/markdown',
)
