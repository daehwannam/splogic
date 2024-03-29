
from dhnamlib.pylib.object import ObjectCache
from dhnamlib.pylib.lazy import LazyProxy


object_cache = ObjectCache
set_tqdm_use_initializer = object_cache.set_initializer


def _no_tqdm(iterator, /, *args, **kwargs):
    return iterator


@LazyProxy
def tqdm():
    if object_cache.get_object():
        from tqdm import tqdm
    else:
        return _no_tqdm
    return tqdm


@LazyProxy
def xtqdm():
    if object_cache.get_object():
        from dhnamlib.pylib.iteration import xtqdm
    else:
        return _no_tqdm
    return xtqdm


@LazyProxy
def utqdm():
    if object_cache.get_object():
        from dhnamlib.pylib.iteration import utqdm
    else:
        return _no_tqdm
    return utqdm
