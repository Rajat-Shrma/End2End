from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    with open(path,'r') as f:
        req=[line.strip() for line in f.readlines()]
        if HYPEN_E_DOT in req:
            req.remove(HYPEN_E_DOT)

    return req

setup(
    name='mlproject',
    version='0.0.1',
    author='Rajat',
    author_email='rajattsharma87077@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
)
