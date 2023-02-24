from setuptools import setup

setup(
    name='mwtoolbox',
    version='0.1.0',    
    description='Microwave Toolbox',
    url='https://github.com/terdol/microwave_toolbox',
    author='Tuncay Erdöl',
    author_email='terdol@hotmail.com',
    license='BSD 2-clause',
    packages=['mwtoolbox'],
    install_requires=['quantities',
                      'numpy',
                      'scipy',
                      'sympy',
                      ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',  
        'Operating System :: POSIX :: Linux', 
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)