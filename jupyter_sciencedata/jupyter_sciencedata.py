# This is a fork of JupyterS3 - https://github.com/uktrade/jupyters3
# modified to use a plain unauthenticated/trusted WebDAV service instead of S3.

import asyncio
import base64
from collections import namedtuple
import datetime
import hashlib
import hmac
import itertools
import json
import mimetypes
import os
import threading
import re
import time
import urllib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tornado import gen
from tornado.httpclient import (
    AsyncHTTPClient,
    HTTPError as HTTPClientError,
    HTTPRequest,
)
from tornado.ioloop import IOLoop
from tornado.locks import Lock
from tornado.web import HTTPError as HTTPServerError

AsyncHTTPClient.configure("tornado.curl_httpclient.CurlAsyncHTTPClient")

from jupyter_server.services.contents.checkpoints import Checkpoints, GenericCheckpointsMixin
from jupyter_server.services.contents.filecheckpoints import GenericFileCheckpoints

from traitlets.config.configurable import Configurable
from traitlets import (
    Dict,
    Unicode,
    Instance,
    TraitType,
    TraitError,
    Type,
    default,
    validate,
    HasTraits,
)

import nbformat
from nbformat.v4 import new_notebook
from jupyter_server.services.contents.manager import (
    ContentsManager,
)

from webdav3.client import Client
from webdav3.exceptions import RemoteResourceNotFound

from jupyter_server.transutils import _i18n
from jupyter_client.utils import run_sync

NOTEBOOK_SUFFIX = '.ipynb'
CHECKPOINT_SUFFIX = '.checkpoints'
UNTITLED_NOTEBOOK = 'Untitled'
UNTITLED_FILE = 'Untitled'
UNTITLED_DIRECTORY  = 'Untitled Folder'

Context = namedtuple('Context', [
    'logger', 'multipart_uploads'
])

CheckpointContext = namedtuple('Context', [
    'logger'
])

SCIENCEDATA_HEADERS = {};
SCIENCEDATA_PREFIX = "/files";
server_root = os.getenv('JUPYTER_SERVER_ROOT')
if server_root != None:
    SCIENCEDATA_PREFIX = SCIENCEDATA_PREFIX + "/" + server_root.strip("/")
SCIENCEDATA_HOST = "sciencedata";

webdav_options = {
 'webdav_hostname': "https://" + SCIENCEDATA_HOST + SCIENCEDATA_PREFIX,
 'webdav_login': '',
 'webdav_password': '',
 'verify': False
}
webdav_client = Client(webdav_options)
webdav_client.default_options['SSL_VERIFYPEER'] = False 
webdav_client.default_options['SSL_VERIFYHOST'] = False
webdav_client.verify = False

# As far as I can see from
# https://github.com/ezhov-evgeny/webdav-client-python-3/blob/871ea5f9b862553465551dd79dd5b6b298e3ff17/webdav3/client.py
# there's no way of setting headers with webdav-client, so not much point in setting some for non-webdav methods.

class ExpiringDict:

    def __init__(self, seconds):
        self._seconds = seconds
        self._store = {}

    def _remove_old_keys(self, now):
        self._store = {
            key: (expires, value)
            for key, (expires, value) in self._store.items()
            if expires > now
        }

    def __getitem__(self, key):
        now = int(time.time())
        self._remove_old_keys(now)
        return self._store[key][1]

    def __setitem__(self, key, value):
        now = int(time.time())
        self._remove_old_keys(now)
        self._store[key] = (now + self._seconds, value)

    def __delitem__(self, key):
        now = int(time.time())
        self._remove_old_keys(now)
        del self._store[key]

class NoOpCheckpoints(GenericCheckpointsMixin, Checkpoints):
    """requires the following methods:"""
    def create_file_checkpoint(self, content, format, path):
        """ -> checkpoint model"""
    def create_notebook_checkpoint(self, nb, path):
        """ -> checkpoint model"""
    def get_file_checkpoint(self, checkpoint_id, path):
        """ -> {'type': 'file', 'content': <str>, 'format': {'text', 'base64'}}"""
    def get_notebook_checkpoint(self, checkpoint_id, path):
        """ -> {'type': 'notebook', 'content': <output of nbformat.read>}"""
    def delete_checkpoint(self, checkpoint_id, path):
        """deletes a checkpoint for a file"""
    def list_checkpoints(self, path):
        """returns a list of checkpoint models for a given file,
        default just does one per file
        """
        return []
    def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        """renames checkpoint from old path to new path"""

class OpCheckpoints(GenericCheckpointsMixin, Checkpoints):

    def _context(self):
        return CheckpointContext(
            logger=self.log,
        )

    def create_file_checkpoint(self, content, format, path):
        @gen.coroutine
        def create_checkpoint_async():
            return (yield _create_checkpoint(self._context(), path))

        return _run_sync_in_new_thread(create_checkpoint_async)

    def create_notebook_checkpoint(self, nb, path):
        @gen.coroutine
        def create_checkpoint_async():
            return (yield _create_checkpoint(self._context(), path))

        return _run_sync_in_new_thread(create_checkpoint_async)

    def get_file_checkpoint(self, checkpoint_id, path):
        @gen.coroutine
        def get_file_checkpoint_async():
             return (yield _get_model_at_checkpoint(self._context(), 'file', checkpoint_id, path))

        return _run_sync_in_new_thread(get_file_checkpoint_async)

    def get_notebook_checkpoint(self, checkpoint_id, path):
        @gen.coroutine
        def get_notebook_checkpoint_async():
             return (yield _get_model_at_checkpoint(self._context(), 'notebook', checkpoint_id, path))

        return _run_sync_in_new_thread(get_notebook_checkpoint_async)

    def delete_checkpoint(self, checkpoint_id, path):
        self._context().logger.info('Deleting checkpoint')

        @gen.coroutine
        def delete_checkpoint_async():
            return (yield _delete_checkpoint(self._context(), checkpoint_id, path))

        return _run_sync_in_new_thread(delete_checkpoint_async)

    def list_checkpoints(self, path):
        self._context().logger.info('Listing checkpoints at '+path)
        
        @gen.coroutine
        def list_checkpoints_async():
            return (yield _list_checkpoints(self._context(), path))

        return _run_sync_in_new_thread(list_checkpoints_async)

    def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        @gen.coroutine
        def rename_checkpoint_async():
             return (yield _rename_checkpoint(self._context(), checkpoint_id, old_path, new_path))

        return _run_sync_in_new_thread(rename_checkpoint_async)

#     @gen.coroutine
#     def restore_checkpoint(self, checkpoint_id, path):
#         with (yield self.write_lock.acquire()):
#             return (yield _restore_checkpoint(self._context(), checkpoint_id, path))

def _checkpoint_path(path, checkpoint_id):
    dir_path = os.path.dirname(path)
    return dir_path + '/' + CHECKPOINT_SUFFIX + '/' + checkpoint_id

@gen.coroutine
def _create_checkpoint(context, path):
    model = yield _get(context, path, content=True, type=None, format=None)
    type = model['type']
    content = model['content']
    format = model['format']
 
    checkpoint_id = str(int(time.time() * 1000000))
    checkpoint_path = _checkpoint_path(path, checkpoint_id)
    checkpoint_path = '/' + checkpoint_path.lstrip('/')
    if not (yield _dir_exists(context, os.path.dirname(checkpoint_path))):
        webdav_client.mkdir(os.path.dirname(checkpoint_path))
    context.logger.info("Saving checkpoint "+type+":"+format+":"+checkpoint_path+":"+checkpoint_id)
    yield SAVERS[(type, format)](context, None, content, checkpoint_path)
    # This is a new object, so shouldn't be any eventual consistency issues
    checkpoint = yield GETTERS[(type, format)](context, checkpoint_path, False)
    return {
        'id': checkpoint_id,
        'last_modified': checkpoint['last_modified'],
    }
 
@gen.coroutine
def _get_model_at_checkpoint(context, type, checkpoint_id, path):
    format = _format_from_type_and_path(context, type, path)
    checkpoint_path = _checkpoint_path(path, checkpoint_id)
    return (yield GETTERS[(type, format)](context, checkpoint_path, True))
 
#@gen.coroutine
#def _restore_checkpoint(context, checkpoint_id, path):
#    type = (yield _get(context, path, content=False, type=None, format=None))['type']
#    model = yield _get_model_at_checkpoint(context, type, checkpoint_id, path)
#    yield _save(context, model, path)

@gen.coroutine
def _rename_checkpoint(context, checkpoint_id, old_path, new_path):
    old_checkpoint_path = _checkpoint_path(old_path, checkpoint_id)
    new_checkpoint_path = _checkpoint_path(new_path, checkpoint_id)
    yield _rename(context, old_checkpoint_path, new_checkpoint_path)

@gen.coroutine
def _delete_checkpoint(context, checkpoint_id, path):
    checkpoint_path = _checkpoint_path(path, checkpoint_id)
    yield _delete(context, checkpoint_path)

@gen.coroutine
def _list_checkpoints(context, path):
    checkpoints_dir = os.path.dirname(path) + '/' + CHECKPOINT_SUFFIX

    try:
        files = webdav_client.list(checkpoints_dir, get_info=True)
    except RemoteResourceNotFound:
        files = []
    
    files = list(filter(lambda file: not file['path'].rstrip('/').endswith(CHECKPOINT_SUFFIX), files))
    
    return [
        {
            'id': file['path'][(file['path'].rfind('/' + CHECKPOINT_SUFFIX + '/') + len('/' + CHECKPOINT_SUFFIX + '/')):],
            'last_modified': file['modified'],
        }
        for file in files
    ]

class JupyterScienceData(ContentsManager, HasTraits):

    checkpoints_class = OpCheckpoints
    #checkpoints_class = NoOpCheckpoints

    root_dir = Unicode("/", config=True)

    preferred_dir = Unicode(
        "",
        config=True,
        help=_i18n(
            "Preferred starting directory to use for notebooks. This is an API path (`/` separated, relative to root dir)"
        ),
    )

    @validate("preferred_dir")
    def _validate_preferred_dir(self, proposal):
        value = proposal["value"].strip("/")
        try:
            import inspect

            if inspect.iscoroutinefunction(self.dir_exists):
                dir_exists = run_sync(self.dir_exists)(value)
            else:
                dir_exists = self.dir_exists(value)
        except HTTPServerError as e:
            raise TraitError(e.log_message) from e
        if not dir_exists:
            raise TraitError(_i18n("Preferred directory not found: %r") % value)
        try:
            if value != self.parent.preferred_dir:
                self.parent.preferred_dir = os.path.join(self.root_dir, *value.split("/"))
        except (AttributeError, TraitError):
            pass
        return value

    multipart_uploads = Instance(ExpiringDict)

    @default('multipart_uploads')
    def _multipart_uploads_default(self):
        return ExpiringDict(60 * 60 * 1)

    def is_hidden(self, path):
        return False

    # The next functions are not expected to be coroutines
    # or return futures. They have to block the event loop.

    def dir_exists(self, path):

        @gen.coroutine
        def dir_exists_async():
            return (yield _dir_exists(self._context(), path))

        return _run_sync_in_new_thread(dir_exists_async)

    def file_exists(self, path):

        @gen.coroutine
        def file_exists_async():
            return (yield _file_exists(self._context(), path))

        return _run_sync_in_new_thread(file_exists_async)

    def get(self, path, content=True, type=None, format=None, require_hash=True):

        @gen.coroutine
        def get_async():
            return (yield _get(self._context(), path, content, type, format))

        return _run_sync_in_new_thread(get_async)

    def save(self, model, path):
        @gen.coroutine
        def save_async():
            return (yield _save(self._context(), model, path))

        return _run_sync_in_new_thread(save_async)

    def delete_file(self, path):
        self._context().logger.info('Deleting file ' + path)
        @gen.coroutine
        def delete_async():
            return (yield _delete(self._context(), path))

        return _run_sync_in_new_thread(delete_async)

    def delete(self, path):
        self._context().logger.info('Deleting ' + path)
        @gen.coroutine
        def delete_async():
            return (yield _delete(self._context(), path))

        return _run_sync_in_new_thread(delete_async)


    def rename_file(self, old_path, new_path):
        @gen.coroutine
        def rename_async():
            return (yield _rename(self._context(), old_path, new_path))

        return _run_sync_in_new_thread(rename_async)

    @gen.coroutine
    def new_untitled(self, path='', type='', ext=''):
        return (yield _new_untitled(self._context(), path, type, ext))

    @gen.coroutine
    def new(self, model, path):
        return (yield _new(self._context(), model, path))

#    @gen.coroutine
#    def copy(self, from_path, to_path):
#        return (yield _copy(self._context(), from_path, to_path))

    def _context(self):
        return Context(
            logger=self.log,
            multipart_uploads=self.multipart_uploads,
        )

def _final_path_component(path):
    return (re.sub('/$', '', path)).split('/')[-1]

# We don't save type/format to sciencedata, so we do some educated guesswork
# as to the types/formats of returned values.
@gen.coroutine
def _type_from_path(context, path):
    type = \
        'notebook' if path.endswith(NOTEBOOK_SUFFIX) else \
        'directory' if _is_root(path) or (yield _dir_exists(context, path)) else \
        'file'
    return type

def _format_from_type_and_path(context, type, path):
    type = \
        'json' if type == 'notebook' else \
        'json' if type == 'directory' else \
        'text' if (mimetypes.guess_type(path)[0] == 'text/plain' or mimetypes.guess_type(path)[0] == 'text/markdown') else \
        'base64'
    return type

def _type_from_path_not_directory(path):
    type = \
        'notebook' if path.endswith(NOTEBOOK_SUFFIX) else \
        'file'
    return type

def _is_root(path):
    is_notebook_root = path == ''
    is_lab_root = path == '/'
    return is_notebook_root or is_lab_root

@gen.coroutine
def _dir_exists(context, path):
    if _is_root(path):
        return True
    try:
        etag = _get_etag(context, path)
    except HTTPServerError as exception:
        etag = 'notfound'
    if etag=='':
        context.logger.info('Dir exists')
        return True
    return False

@gen.coroutine
def _file_exists(context, path):
    if _is_root(path):
        return False
    try:
        etag = _get_etag(context, path)
    except HTTPServerError as exception:
        etag = ''
    if etag!='':
        context.logger.info('File exists')
        return True
    return False

def _get_etag(context, path):

    @gen.coroutine
    def _get_etag_async():
        if _is_root(path):
            return ''
        response = yield _make_sciencedata_http_request(context, 'HEAD', path, {}, b'', {})
        etag = response.headers['ETag'] if ('ETag' in response.headers) else ''
        if etag:
            context.logger.info('ETag: '+etag)
        else:
            context.logger.info('Headers: '+':'.join(response.headers))
        return etag

    return _run_sync_in_new_thread(_get_etag_async)

@gen.coroutine
def _exists(context, path):
    return (yield _file_exists(context, path)) or (yield _dir_exists(context, path)) 

@gen.coroutine
def _get(context, path, content, type, format):
    type_to_get = type if type is not None else (yield _type_from_path(context, path))
    format_to_get = format if format is not None else _format_from_type_and_path(context, type_to_get, path)
    return (yield GETTERS[(type_to_get, format_to_get)](context, path, content))

# Backwards compatibility
# function that changes lists to strings
# FO: fix - don't strip source - indents matter
def fix_json(myjson):
    if myjson is None or 'fixed' in myjson:
        return
    if 'worksheets' in myjson and not myjson['worksheets'] is None:
        for worksheet in myjson['worksheets']:
            fix_json_cells(worksheet)
    elif 'cells' in myjson:
        fix_json_cells(myjson)
    myjson['fixed'] = True

def fix_json_cells(j):
    if not 'cells' in j or j['cells'] is None:
        return
    for cell in j['cells']:
        if 'text' in cell and type(cell['text']) == list:
            cell['text'] = "".join([l.strip() for l in cell['text']])
        elif 'source' in cell and type(cell['source']) == list:
            cell['source'] = "".join([l for l in cell['source']])
        if 'outputs' in cell and not cell['outputs'] is None:
            for k in range(len(cell['outputs'])):
                if 'text' in cell['outputs'][k] and type(cell['outputs'][k]['text']) == list:
                    cell['outputs'][k]['text'] = "\n".join([l.strip() for l in cell['outputs'][k]['text']])

@gen.coroutine
def _get_notebook(context, path, content):
    # context, path, content, type, mimetype, format, decode
    notebook_dict = yield _get_any(context, path, content, 'notebook', None, 'json', lambda file_bytes: json.loads(file_bytes.decode('utf-8')))
    try:
        fix_json(notebook_dict['content'])
    except Exception as e:
        context.logger.error('Notebook fixing failed, '+str(e))
    notebook_dict['mimetype'] = 'application/x-ipynb+json'
    ret = nbformat.from_dict(notebook_dict)
    return ret

@gen.coroutine
def _get_file_base64(context, path, content):
    return (yield _get_any(context, path, content, 'file', 'application/octet-stream', 'base64', lambda file_bytes: base64.b64encode(file_bytes).decode('utf-8')))

@gen.coroutine
def _get_file_text(context, path, content):
    return (yield _get_any(context, path, content, 'file', 'text/plain', 'text', lambda file_bytes: file_bytes.decode('utf-8')))

@gen.coroutine
def _get_any(context, path, content, type, mimetype, format, decode):
    #method = 'GET' if content else 'HEAD'
    # We need to get the body, even for content=0 requests, as these serve to check if a file has changed and need the md5 hash
    method = 'GET'
    response = yield _make_sciencedata_http_request(context, method, path, {}, b'', {})
    file_bytes = response.body
    last_modified_str = response.headers['Last-Modified']
    last_modified = datetime.datetime.strptime(last_modified_str, "%a, %d %b %Y %H:%M:%S GMT")
    return {
        'name': _final_path_component(path),
        'path': path,
        'type': type,
        'mimetype': mimetype,
        'writable': True,
        'last_modified': last_modified, 
        'created': last_modified,
        'format': format if content else None,  
        'content': decode(file_bytes) if content else None,
        'hash_algorithm': 'md5',
        'hash': hashlib.md5(file_bytes).hexdigest()
    }


@gen.coroutine
def _get_directory(context, path, content):
    files = webdav_client.list(path, get_info=True) if content else []
    if(len(files)>0):
        files.pop(0)
    return {
        'name': _final_path_component(path),
        'path': path.replace(SCIENCEDATA_PREFIX, '', 1),
        'type': 'directory',
        'mimetype': None,
        'writable': True,
        'last_modified': datetime.datetime.fromtimestamp(86400), 
        'created': datetime.datetime.fromtimestamp(86400),
        'format': 'json' if content else None,
        'content': [
            {
                'type': 'directory' if file['isdir'] else _type_from_path_not_directory(file['path']),
                'name': _final_path_component(file['path']),
                'path': file['path'].replace(SCIENCEDATA_PREFIX, '', 1).rstrip('/'),
                'last_modified': file['modified'],
            }
            for (file) in files
        ] if content else None
    }

@gen.coroutine
def _save(context, model, path):
    type_to_save = model['type'] if 'type' in model else (yield _type_from_path(context, path))
    format_to_save = model['format'] if 'format' in model else _format_from_type_and_path(context, type_to_save, path)
    return (yield SAVERS[(type_to_save, format_to_save)](
        context,
        model['chunk'] if 'chunk' in model else None,
        model['content'] if 'content' in model else None,
        path,
    ))

@gen.coroutine
def _save_notebook(context, chunk, content, path):
    return (yield _save_any(context, chunk, json.dumps(content).encode('utf-8'), path, 'notebook', None))

@gen.coroutine
def _save_file_base64(context, chunk, content, path):
    return (yield _save_any(context, chunk, base64.b64decode(content.encode('utf-8')), path, 'file', 'application/octet-stream'))

@gen.coroutine
def _save_file_text(context, chunk, content, path):
    return (yield _save_any(context, chunk, content.encode('utf-8'), path, 'file', 'text/plain'))

@gen.coroutine
def _save_directory(context, chunk, content, path):
    if(not webdav_client.mkdir(path)):
        raise Exception("something went wrong...")
    return _saved_model(path, 'directory', None, datetime.datetime.now())

@gen.coroutine
def _save_any(context, chunk, content_bytes, path, type, mimetype):
    response = \
        (yield _save_bytes(context, content_bytes, path, type, mimetype)) if chunk is None else \
        (yield _save_chunk(context, chunk, content_bytes, path, type, mimetype))

    return response

@gen.coroutine
def _save_chunk(context, chunk, content_bytes, path, type, mimetype):
    # Chunks are 1-indexed
    if chunk == 1:
        context.multipart_uploads[path] = []
    context.multipart_uploads[path].append(content_bytes)

    # -1 is the last chunk
    if chunk == -1:
        combined_bytes = b''.join(context.multipart_uploads[path])
        del context.multipart_uploads[path]
        return (yield _save_bytes(context, combined_bytes, path, type, mimetype))
    else:
        return _saved_model(path, type, mimetype, datetime.datetime.now())

@gen.coroutine
def _save_bytes(context, content_bytes, path, type, mimetype):
    response = yield _make_sciencedata_http_request(context, 'PUT', path, {}, content_bytes, {})

    last_modified_str = response.headers['Date']
    last_modified = datetime.datetime.strptime(last_modified_str, "%a, %d %b %Y %H:%M:%S GMT")
    return _saved_model(path, type, mimetype, last_modified)

def _saved_model(path, type, mimetype, last_modified):
    return {
        'name': _final_path_component(path),
        'path': path,
        'type': type,
        'mimetype': mimetype,
        'writable': True,
        'last_modified': last_modified, 
        'created': last_modified,
        'format': None,
        'content': None,
    }

@gen.coroutine
def _increment_filename(context, filename, path='', insert=''):
    basename, dot, ext = filename.partition('.')
    suffix = dot + ext

    for i in itertools.count():
        insert_i = f'{insert}{i}' if i else ''
        name = f'{basename}{insert_i}{suffix}'
        if not (yield _exists(context, f'/{path}/{name}')):
            break
    return name

@gen.coroutine
def _rename(context, old_path, new_path):
    #if (not (yield _exists(context, old_path))):
    #    raise HTTPServerError(400, "Source does not exist")

    #if (yield _exists(context, new_path)):
    #    raise HTTPServerError(400, "Target already exists")
    type = yield _type_from_path(context, old_path)

    if old_path == new_path :
        return _saved_model(new_path, type, None, datetime.datetime.now())

    # webdav_client returns nothing. We need headers
    #response = yield webdav_client.move(remote_path_from=old_path, remote_path_to=new_path)
    encoded_new_path = urllib.parse.quote(SCIENCEDATA_PREFIX+new_path, safe='/~')
    new_url = f'https://{SCIENCEDATA_HOST}{encoded_new_path}'
    response = yield _make_sciencedata_http_request(context, 'MOVE', old_path, {}, b'', {'Destination':new_url})
    last_modified_str = response.headers['Date']
    last_modified = datetime.datetime.strptime(last_modified_str, "%a, %d %b %Y %H:%M:%S GMT")
    mimestring = response.headers['Content-Type']
    mimesplit = mimestring.split(";", 1)
    mimetype = mimesplit[0]
    return _saved_model(new_path, type, mimetype, last_modified)

@gen.coroutine
def _delete(context, path):
    return (yield _make_sciencedata_http_request(context, 'DELETE', path, {}, b'', {}))

@gen.coroutine
def _new_untitled(context, path, type, ext):
    if not (yield _dir_exists(context, path)):
        raise HTTPServerError(404, 'No such directory: %s' % path)

    model_type = \
        type if type else \
        'notebook' if ext == '.ipynb' else \
        'file'

    untitled = \
        UNTITLED_DIRECTORY if model_type == 'directory' else \
        UNTITLED_NOTEBOOK if model_type == 'notebook' else \
        UNTITLED_FILE
    insert = \
        ' ' if model_type == 'directory' else \
        ''
    ext = \
        '.ipynb' if model_type == 'notebook' else \
        ext

    name = yield _increment_filename(context, untitled + ext, path, insert=insert)
    path = u'{0}/{1}'.format(path, name)

    model = {
        'type': model_type,
    }
    return (yield _new(context, model, path))

@gen.coroutine
def _new(context, model, path):
    if model is None:
        model = {}

    model.setdefault('type', 'notebook' if path.endswith('.ipynb') else 'file')

    if 'content' not in model and model['type'] == 'notebook':
        model['content'] = new_notebook()
        model['format'] = 'json'
    elif 'content' not in model and model['type'] == 'file':
        model['content'] = ''
        model['format'] = 'text'

    return (yield _save(context, model, path))

@gen.coroutine
def _copy(context, from_path, to_path):
    model = yield _get(context, from_path, content=False, type=None, format=None)
    if model['type'] == 'directory':
        raise HTTPServerError(400, "Can't copy directories")

    from_dir, from_name = \
        from_path.rsplit('/', 1) if '/' in from_path else \
        ('', from_path)

    to_path = \
        to_path if to_path is not None else \
        from_dir

    if (yield _dir_exists(context, to_path)):
        copy_pat = re.compile(r'\-Copy\d*\.')
        name = copy_pat.sub(u'.', from_name)
        to_name = yield _increment_filename(context, name, to_path, insert='-Copy')
        to_path = u'{0}/{1}'.format(to_path, to_name)

    return {
        **model,
        'name': to_name,
        'path': to_path,
    }

@gen.coroutine
def _make_sciencedata_http_request(context, method, path, query, payload, headers):
    all_headers = {**SCIENCEDATA_HEADERS, **headers}
    querystring = urllib.parse.urlencode(query, safe='~', quote_via=urllib.parse.quote)
    encoded_path = urllib.parse.quote(SCIENCEDATA_PREFIX+path, safe='/~')
    url = f'https://{SCIENCEDATA_HOST}{encoded_path}' + (('?' + querystring) if querystring else '')

    body = \
        payload if method == 'PUT' else \
        None
    request = HTTPRequest(url, method=method, headers=all_headers, body=body, validate_cert=False, allow_nonstandard_methods=True)

    try:
        context.logger.info('Running HTTP request '+method+' on '+url)
        response = (yield AsyncHTTPClient().fetch(request))
        #IOLoop.current().start()
    except HTTPClientError as exception:
        if not hasattr(exception.response, 'code'):
            context.logger.warning('No response')
            raise HTTPServerError(0, 'Error accessing '+url)
        elif exception.response.code != 404:
            context.logger.warning(exception.response.body)
        raise HTTPServerError(exception.response.code, 'Error accessing '+url)

    return response

def _run_sync_in_new_thread(func):
    result = None
    exception = None
    def _func():
        nonlocal result
        nonlocal exception
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            result = IOLoop.current().run_sync(func)
        except BaseException as _exception:
            exception = _exception

    thread = threading.Thread(target=_func)
    thread.start()
    thread.join()

    if exception is not None:
        raise exception
    else:
        return result


GETTERS = {
    ('notebook', 'json'): _get_notebook,
    ('file', 'text'): _get_file_text,
    ('file', 'base64'):  _get_file_base64,
    ('directory', 'json'): _get_directory,
}


SAVERS = {
    ('notebook', 'json'): _save_notebook,
    ('file', 'text'): _save_file_text,
    ('file', 'base64'):  _save_file_base64,
    ('directory', 'json'): _save_directory,
}
