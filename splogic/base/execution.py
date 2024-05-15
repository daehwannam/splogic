
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
class ExprCompiler(Compiler):
    # interface = Interface(Compiler)

    @implement
    def compile_tree(self, tree, tolerant=False):
        return tree.get_expr_str()


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


# NO_DENOTATION = object()
NO_DENOTATION = 'NO_DENOTATION'


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


class ExecResult(metaclass=ABCMeta):
    '''
    The execution result that is returned from `Executor.execute`.
    '''

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass


class Executor(metaclass=ABCMeta):
    @abstractmethod
    def execute(self, programs, contexts) -> ExecResult:
        pass

    # @abstractmethod
    # def wait_until_all_done():
    #     pass


@subclass
class InstantExecResult(ExecResult):
    # interface = Interface(ExecResult)

    def __init__(self, values):
        # self._values = self.post_process(value)
        self._values = values

    @implement
    def get(self):
        return self._values

    @implement
    def is_done(self) -> bool:
        return True

    # def post_process(self, value):
    #     return value


@subclass
class InstantExecutor(Executor):
    # interface = Interface(Executor)

    def __init__(
            self,
            result_cls=InstantExecResult,
            context_wrapper=identity,
    ):
        self.result_cls = result_cls
        self.context_wrapper = context_wrapper

    @implement
    def execute(self, programs, contexts) -> ExecResult:
        assert len(programs) == len(contexts)
        result = self.result_cls(
            tuple(program(self.context_wrapper(context))
                  for program, context in zip(programs, contexts)))

        return result


@subclass
class LazyExecResult(ExecResult):
    # interface = Interface(ExecResult)

    def __init__(self, lazy_executor):
        self._lazy_executor = lazy_executor

    def _set_values(self, values):
        self._values = values

    @implement
    def get(self):
        self._lazy_executor._work()
        return self._values

    @implement
    def is_done(self) -> bool:
        return hasattr(self, '_values')

    # def post_process(self, value):
    #     return value


@subclass
class LazyExecutor(Executor):
    # interface = Interface(Executor)

    def __init__(
            self,
            result_cls=LazyExecResult,
            context_wrapper=identity,
    ):
        self.result_cls = result_cls
        self.context_wrapper = context_wrapper

        self._postponed_batch_groups = []

    @implement
    def execute(self, programs, contexts) -> ExecResult:
        assert len(programs) == len(contexts)
        lazy_result = LazyExecResult(self)
        self._postponed_batch_groups.append((lazy_result, programs, contexts))
        return lazy_result

    def _work(self):
        self._process()
        self._postponed_batch_groups = []

    def _process(self):
        for lazy_result, programs, contexts in self._postponed_batch_groups:
            lazy_result._set_values(
                tuple(program(self.context_wrapper(context))
                      for program, context in zip(programs, contexts)))


@notimplemented
class AsyncExecutor(Executor):
    pass


@notimplemented
class ContextCreater(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
