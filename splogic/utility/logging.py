
from dhnamlib.pylib.object import ObjectCache
from dhnamlib.pylib.lazy import LazyProxy
from dhnamlib.pylib.filesys import make_logger, NoLogger


object_cache = ObjectCache()

set_logger_initializer = object_cache.set_initializer

logger = LazyProxy(object_cache.get_object)
