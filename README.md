# JupyterScienceData

Jupyter Notebook Contents Manager for ScienceData, https://sciencedata.dk/ - based on jupyters3 by Michael Charemza, https://github.com/uktrade/jupyters3

## Installation

```
pip install jupyter_sciencedata/
```

## Configuration

Add the following to ~/.jupyter/jupyter_notebook_config.py

```
from jupyter_sciencedata import JupyterScienceData
c = get_config()
c.NotebookApp.contents_manager_class = JupyterScienceData
```