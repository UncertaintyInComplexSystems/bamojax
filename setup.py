from setuptools import setup, find_packages
import versioneer

setup(
    name='bamojax',  
    description='Bayesian Modelling with JAX',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",  
    url="https://github.com/UncertaintyInComplexSystems/bamojax",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),  # Automatically discover all packages
    install_requires=[ 
        'jax>=0.4.33', 
        'jaxlib>=0.4.33',
        'blackjax>=1.2.4',
        'numpyro',
        'jaxtyping>=0.2.34'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Update based on your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',  
)
