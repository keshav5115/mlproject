from setuptools import find_packages,setup



HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)-> list[str]:

    '''
    This function returns a list of requirements
    '''
    
    with open(file_path,'r') as file_obj:
        requirements=file_obj.readlines()
        requirements=[val.replace('\n','') for val in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name = 'Mlproject',
    version='0.0.0',
    author='keshav',
    author_email='keshav@gmail.com',
    packages=find_packages(),
    install_requires =get_requirements('requirements.txt')
)