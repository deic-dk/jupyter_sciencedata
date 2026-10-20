import os

if os.getenv('SCIENCEDATA_GUEST'):
    # Guest (anonymous) pod, started by run_pod with SCIENCEDATA_GUEST=1: there is
    # no home server, the notebook lives in a local work directory and nothing is
    # kept. The images' Jupyter config names jupyter_sciencedata.JupyterScienceData
    # as the contents manager, so hand out the stock file manager under that name
    # and never touch the WebDAV client.
    try:
        from jupyter_server.services.contents.largefilemanager import LargeFileManager as JupyterScienceData
    except ImportError:  # notebook < 7 without jupyter_server
        from notebook.services.contents.largefilemanager import LargeFileManager as JupyterScienceData
else:
    from .jupyter_sciencedata import JupyterScienceData
