from setuptools import setup, find_packages
import versioneer

setup(
    name='bamojax',  
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),  # Automatically discover all packages
    install_requires=[ 
        'jax>=0.4.35<0.5.0', 
        'jaxlib>=0.4.34<0.5.0',
        'blackjax==1.2.4',
        'distrax==0.1.5',
        'jaxtyping>=0.2.34'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Update based on your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',  
)
