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

from tornado import gen
from tornado.httpclient import (
    AsyncHTTPClient,
    HTTPError as HTTPClientError,
    HTTPRequest,
)
from tornado.ioloop import IOLoop
from tornado.locks import Lock
from tornado.web import HTTPError as HTTPServerError
from traitlets.config.configurable import Configurable
from traitlets import (
    Dict,
    Unicode,
    Instance,
    TraitType,
    Type,
    default,
)

import nbformat
from nbformat.v4 import new_notebook
from notebook.services.contents.manager import (
    ContentsManager,
)

from webdav3.client import Client

NOTEBOOK_SUFFIX = '.ipynb'
CHECKPOINT_SUFFIX = '.checkpoints'
UNTITLED_NOTEBOOK = 'Untitled'
UNTITLED_FILE = 'Untitled'
UNTITLED_DIRECTORY  = 'Untitled Folder'

Context = namedtuple('Context', [
    'logger', 'multipart_uploads'
])

SCIENCEDATA_HEADERS = {};
SCIENCEDATA_PREFIX = "/files/";
SCIENCEDATA_HOST = "sciencedata";

webdav_options = {
 'webdav_hostname': "https://" + SCIENCEDATA_HOST + SCIENCEDATA_PREFIX,
 'webdav_login': '',
 'webdav_password': '',
 'verify': False
}
webdav_client = Client(webdav_options)

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


class Datetime(TraitType):
    klass = datetime.datetime
    default_value = datetime.datetime(1900, 1, 1)

class JupyterScienceData(ContentsManager):

    # Do not use a checkpoints class: the rest of the system
    # only expects a ContentsManager
    checkpoints_class = None

    # Some of the write functions contain multiple S3 call
    # We do what we can to prevent bad things from happening
    write_lock = Instance(Lock)

    @default('write_lock')
    def _write_lock_default(self):
        return Lock()

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

    def get(self, path, content=True, type=None, format=None):

        @gen.coroutine
        def get_async():
            return (yield _get(self._context(), path, content, type, format))

        return _run_sync_in_new_thread(get_async)

    def save(self, model, path):
        @gen.coroutine
        def save_async():
            with (yield self.write_lock.acquire()):
                return (yield _save(self._context(), model, path))

        return _run_sync_in_new_thread(save_async)

    def delete_file(self, path):
        @gen.coroutine
        def delete_async():
            with (yield self.write_lock.acquire()):
                return (yield _delete(self._context(), path))

        return _run_sync_in_new_thread(delete_async)

    def rename_file(self, old_path, new_path):
        @gen.coroutine
        def rename_async():
            with (yield self.write_lock.acquire()):
                return (yield _rename(self._context(), old_path, new_path))

        return _run_sync_in_new_thread(rename_async)

    @gen.coroutine
    def new_untitled(self, path='', type='', ext=''):
        with (yield self.write_lock.acquire()):
            return (yield _new_untitled(self._context(), path, type, ext))

    @gen.coroutine
    def new(self, model, path):
        with (yield self.write_lock.acquire()):
            return (yield _new(self._context(), model, path))

    @gen.coroutine
    def copy(self, from_path, to_path):
        with (yield self.write_lock.acquire()):
            return (yield _copy(self._context(), from_path, to_path))

    @gen.coroutine
    def create_checkpoint(self, path):
        with (yield self.write_lock.acquire()):
            return (yield _create_checkpoint(self._context(), path))

    @gen.coroutine
    def restore_checkpoint(self, checkpoint_id, path):
        with (yield self.write_lock.acquire()):
            return (yield _restore_checkpoint(self._context(), checkpoint_id, path))

    @gen.coroutine
    def list_checkpoints(self, path):
        return (yield _list_checkpoints(self._context(), path))

    @gen.coroutine
    def delete_checkpoint(self, checkpoint_id, path):
        with (yield self.write_lock.acquire()):
            return (yield _delete_checkpoint(self._context(), checkpoint_id, path))

    def _context(self):
        return Context(
            logger=self.log,
            multipart_uploads=self.multipart_uploads,
        )

def _final_path_component(path):
    return path.split('/')[-1]

# We don't save type/format to S3, so we do some educated guesswork
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
    @gen.coroutine
    def get_dir_etag():
        try:
            etag = _get_etag(context, path)
        except HTTPClientError as exception:
            if exception.response.code != 404:
                raise HTTPServerError(exception.response.code, 'Error checking if file exists')
            etag = 'notfound'
        return etag

    return (True if _is_root(path) else (True if (yield get_dir_etag())=='' else False))

@gen.coroutine
def _file_exists(context, path):
    if _is_root(path):
        return False
    try:
        etag = _get_etag(context, path)
    except HTTPError as exception:
        if exception.response.code != 404:
            raise HTTPServerError(exception.response.code, 'Error checking if file exists')
        etag = ''
    return etag!=''

def _get_etag(context, path):

    @gen.coroutine
    def _get_etag_async():
        if _is_root(path):
            return ''
        response = yield _make_sciencedata_http_request(context, 'HEAD', path, {}, b'', {})
        etag = response.headers['ETag'] if ('ETag' in response.headers) else ''
        context.logger.warning('ETag: '+etag)
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

@gen.coroutine
def _get_notebook(context, path, content):
    notebook_dict = yield _get_any(context, path, content, 'notebook', None, 'json', lambda file_bytes: json.loads(file_bytes.decode('utf-8')))
    return nbformat.from_dict(notebook_dict)

@gen.coroutine
def _get_file_base64(context, path, content):
    return (yield _get_any(context, path, content, 'file', 'application/octet-stream', 'base64', lambda file_bytes: base64.b64encode(file_bytes).decode('utf-8')))

@gen.coroutine
def _get_file_text(context, path, content):
    return (yield _get_any(context, path, content, 'file', 'text/plain', 'text', lambda file_bytes: file_bytes.decode('utf-8')))

@gen.coroutine
def _get_any(context, path, content, type, mimetype, format, decode):
    method = 'GET' if content else 'HEAD'
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
    }  


@gen.coroutine
def _get_directory(context, path, content):
    files = webdav_client.list(path, get_info=True) if content else []
    return {
        'name': _final_path_component(path),
        'path': path,
        'type': 'directory',
        'mimetype': None,
        'writable': True,
        'last_modified': datetime.datetime.fromtimestamp(86400), 
        'created': datetime.datetime.fromtimestamp(86400),
        'format': 'json' if content else None,
        'content': [
            {
                'type': 'directory' if file['isdir'] else _type_from_path_not_directory(file['path']),
                'name': file['name'],
                'path':file['path'],
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
    return (yield _save_any(context, chunk, b'', path + '/', 'directory', None))

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

def _checkpoint_path(path, checkpoint_id):
    return path + '/' + CHECKPOINT_SUFFIX + '/' + checkpoint_id

@gen.coroutine
def _create_checkpoint(context, path):
    model = yield _get(context, path, content=True, type=None, format=None)
    type = model['type']
    content = model['content']
    format = model['format']

    checkpoint_id = str(int(time.time() * 1000000))
    checkpoint_path = _checkpoint_path(path, checkpoint_id)
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

@gen.coroutine
def _restore_checkpoint(context, checkpoint_id, path):
    type = (yield _get(context, path, content=False, type=None, format=None))['type']
    model = yield _get_model_at_checkpoint(context, type, checkpoint_id, path)
    yield _save(context, model, path)

@gen.coroutine
def _delete_checkpoint(context, checkpoint_id, path):
    checkpoint_path = _checkpoint_path(path, checkpoint_id)
    yield _delete(context, checkpoint_path)

@gen.coroutine
def _list_checkpoints(context, path):
    files = webdav_client.list(path, get_info=True)
    return [
        {
            'id': file['path'][(file['path'].rfind('/' + CHECKPOINT_SUFFIX + '/') + len('/' + CHECKPOINT_SUFFIX + '/')):],
            'last_modified': file['modified'],
        }
        for file in files
    ]

@gen.coroutine
def _rename(context, old_path, new_path):
    if (not (yield _exists(context, old_path))):
        raise HTTPServerError(400, "Source does not exist")

    if (yield _exists(context, new_path)):
        raise HTTPServerError(400, "Target already exists")

    type = yield _type_from_path(context, old_path)
    #response = yield _make_sciencedata_http_request(context, 'MOVE', path, {}, content_bytes, {})
    response = yield webdav_client.move(remote_path_from=path, remote_path_to=new_path)
    last_modified_str = response.headers['Date']
    last_modified = datetime.datetime.strptime(last_modified_str, "%a, %d %b %Y %H:%M:%S GMT")
    mimestring = response.headers['Content-Type']
    mimesplit = mimestring.split(";", 1)
    mimetype = mimesplit[0]
    return _saved_model(new_path, type, mimetype, last_modified)

@gen.coroutine
def _delete(context, path):
    yield _make_sciencedata_http_request(context, 'DELETE', path, {}, b'', {})

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
    request = HTTPRequest(url, method=method, headers=all_headers, body=body, validate_cert=False)

    try:
        context.logger.warning('Running HTTP request '+method+' on '+url)
        response = (yield AsyncHTTPClient().fetch(request))
        #IOLoop.current().start()
    except HTTPClientError as exception:
        if exception.response.code != 404:
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
