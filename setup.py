import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='jupyter_sciencedata',
    version='0.0.2',
    author='Frederik Orellana',
    author_email='frederik@orellana.dk',
    description='Jupyter Notebook Contents Manager for ScienceData - based on jupyters3 by Michael Charemza, https://github.com/uktrade/jupyters3',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/deic-dk/jupyter_sciencedata',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)