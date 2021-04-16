from setuptools import setup

setup(
    name='cottage_analysis',
    version='v0.1',
    packages=['cottage_analysis', 'cottage_analysis.io_module', 'cottage_analysis.io_module.video',
              'cottage_analysis.eye_tracking'],
    url='https://github.com/znamlab/cottage_analysis',
    license='MIT',
    author='Antonin Blot, Yiran He, Petr Znamenskyi',
    author_email='antonin.blot@crick.ac.uk',
    description='Common functions for analysis'
)
