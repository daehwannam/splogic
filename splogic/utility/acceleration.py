
from dhnamlib.pylib.object import ObjectCache
from dhnamlib.pylib.hflib.acceleration import AcceleratorProxy


object_cache = ObjectCache()

set_accelerator_initializer = object_cache.set_initializer

accelerator = AcceleratorProxy(object_cache.get_object)
