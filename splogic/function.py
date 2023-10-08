import functools


class _LogicalFormFunction:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


lf_function = _LogicalFormFunction


class CallLimitExceeded(Exception):
    pass


def make_call_limited(max_num_calls):
    num_calls = 0

    class CallLimited:
        def __init__(self, func):
            functools.update_wrapper(self, func)
            self.func = func

        def __call__(self, *args, **kwargs):
            nonlocal num_calls
            num_calls += 1
            if num_calls > max_num_calls:
                raise CallLimitExceeded
            return self.func(*args, **kwargs)

    return CallLimited


def get_call_limited_namespace(ns, max_num_calls):
    call_limited = make_call_limited(max_num_calls)

    def get_kv_gen():
        for k, v in ns.items():
            new_v = call_limited(v) if isinstance(v, _LogicalFormFunction) else v
            yield k, new_v

    return dict(get_kv_gen())
