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

## Guest pods

With `SCIENCEDATA_GUEST` set in the environment (run_pod sets it for anonymous
"Run this notebook" pods) the package exports Jupyter's stock `LargeFileManager`
under the name `JupyterScienceData`, so the same image configuration gives a
pod without a home server: files are local to the pod and nothing is saved.
`HOME_SERVER_PROTOCOL` (default `https`) selects the scheme towards the home
server for ordinary pods.
