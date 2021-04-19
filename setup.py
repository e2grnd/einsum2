from distutils.core import setup
from os.path import dirname, join
from setuptools import find_packages
import pip
import subprocess

thisDirName = dirname(__file__)
requirementsPath = join(thisDirName, 'requirements.txt')

with open(requirementsPath) as requirementsFile:
    required = requirementsFile.read().splitlines()
 
subprocess.call(["pip", "install","-r",requirementsPath])

setup(
        name='EINCORE',
        version='0.0.1',
        description='Eincore',
        author='Donald Brown',
        author_email='dbrown@e2g.com',
        packages=find_packages())