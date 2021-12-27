from setuptools import setup

setup(
    name='polans',
    version='0.1.0',
    py_modules=['polans'],
    install_requires=[
        'Click',
        'obspy',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'polans = polans:main',
        ],
    },
)