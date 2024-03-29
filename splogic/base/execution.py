
from abc import ABCMeta, abstractmethod

from .grammar import get_extra_ns

# from dhnamlib.pylib.klass import Interface, abstractfunction
from dhnamlib.pylib.klass import subclass, implement
from dhnamlib.pylib.decoration import excepting, notimplemented
from dhnamlib.pylib.function import identity

from dhnamlib.hissplib.macro import prelude
from dhnamlib.hissplib.compile import eval_lissp
from dhnamlib.hissplib.operation import import_operators


prelude()  # used for eval_lissp
import_operators()  # used for eval_lissp


class Compiler(metaclass=ABCMeta):
    @abstractmethod
    def compile_tree(self, tree):
        '''Convert a tree to an executable'''


@subclass
class LispCompiler(Compiler):
    # interface = Interface(Compiler)

    def __init__(self, bindings):
        self.extra_ns = get_extra_ns(bindings)

    @implement
    def compile_tree(self, tree, tolerant=False):
        program = eval_lissp(tree.get_expr_str(), extra_ns=self.extra_ns)
        if tolerant:
            return excepting(Exception, default_fn=runtime_exception_handler)(program)
        else:
            return program


NO_DENOTATION = object()


def runtime_exception_handler(context):
    return NO_DENOTATION


class _InvalidProgram:
    '''
    The instance of this class is used as the output program of parsing
    when the final paring states are incomplete.
    '''

    def __call__(self, context):
        return NO_DENOTATION


INVALID_PROGRAM = _InvalidProgram()


class Result(metaclass=ABCMeta):
    '''
    The execution result that is returned from `Executor.execute_batch`.
    '''

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass


@subclass
class InstantResult(Result):
    # interface = Interface(Result)

    def __init__(self, value):
        # self._value = self.post_process(value)
        self._value = value

    @implement
    def get(self):
        return self._value

    @implement
    def is_done(self) -> bool:
        return True

    # def post_process(self, value):
    #     return value


class Executor(metaclass=ABCMeta):
    @abstractmethod
    def execute_batch(programs, contexts) -> Result:
        pass

    @abstractmethod
    def wait_until_all_done():
        pass


@subclass
class InstantExecutor(Executor):
    # interface = Interface(Executor)

    def __init__(
            self,
            result_cls=InstantResult,
            context_wrapper=identity,
    ):
        self.result_cls = InstantResult
        self.context_wrapper = context_wrapper

    @implement
    def execute_batch(self, programs, contexts) -> Result:
        assert len(programs) == len(contexts)
        results = tuple(self.result_cls(program(self.context_wrapper(context)))
                        for program, context in zip(programs, contexts))

        return results

    @implement
    def wait_until_all_done():
        pass


@notimplemented
class AsyncExecutor(Executor):
    pass


class ContextCreater(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
