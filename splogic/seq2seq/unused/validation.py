from fractions import Fraction
# import transformers
import warnings
# from itertools import chain
from abc import ABCMeta, abstractmethod
from functools import cache

from accelerate.utils.operations import gather_object

from dhnamlib.pylib import iteration
from dhnamlib.pylib.iteration import pairs2dicts, not_none_valued_pairs
from dhnamlib.pylib.time import TimeMeasure
from dhnamlib.pylib.text import camel_to_symbol
# from dhnamlib.pylib.lazy import LazyProxy
# from dhnamlib.pylib.klass import Interface
from dhnamlib.pylib.klass import subclass, implement, abstractfunction
from dhnamlib.pylib.torchlib.dnn import unpad_sequence
from dhnamlib.pylib.mllib.learning import get_performance
# from dhnamlib.pylib.torchlib.optimization import get_linear_schedule_with_warmup
from dhnamlib.pylib.structure import XNamespace
from dhnamlib.pylib.hflib.acceleration import alternate_object

# from configuration import config, coc
# from kqapro.evaluate import whether_equal

from splogic.utility.tqdm import xtqdm, utqdm
from splogic.utility.acceleration import accelerator

from . import learning
# from .execution import postprocess_prediction


# class PredictionCollector:
#     def __init__(self, evaluating, whether_equal, num_return_sequences):
#         self.evaluating = evaluating
#         self.whether_equal = whether_equal
#         self.num_return_sequences = num_return_sequences

#         self.predictions = []
#         if evaluating:
#             self.answers = []
#             self.num_correct = 0

#     def collect(self, *, predictions, answers=None):
#         self.predictions.extend(predictions)
#         if self.evaluating:
#             self.answers.extend(answers)
#             self.num_correct += compute_num_correct(
#                 predictions, answers, self.whether_equal,
#                 num_return_sequences=self.num_return_sequences)

#     def get_accuracy(self):
#         assert len(self.predictions) == len(self.answers) * self.num_return_sequences
#         return (self.num_correct / len(self.answers))

#     def get_accuracy_percent(self):
#         return self.get_accuracy() * 100


def derive_name_from_class(base_name, klass):
    _name = klass.__name__
    if _name.endswith(base_name):
        _name = _name[: -len(base_name)]
    _name = camel_to_symbol(_name)
    return _name


class Counter(metaclass=ABCMeta):
    @abstractmethod
    def compute(self, predictions, answers):
        pass

    @abstractfunction
    def compute_oracle(self, predictions, answers, num_return_sequences):
        pass

    @abstractmethod
    def get_name(cls):
        pass


@subclass
class ExampleCounter(Counter):
    # interface = Interface(Counter)

    @implement
    def compute(self, predictions, answers):
        # assert len(predictions) == len(answers)
        return len(answers)

    @implement
    def compute_oracle(self, predictions, answers, num_return_sequences):
        # assert len(predictions) == len(answers) * num_return_sequences
        return len(answers)

    @implement
    def get_name(cls):
        return 'num_examples'


@subclass
class CorrectCounter(Counter):
    # interface = Interface(Counter)

    def __init__(self, whether_equal):
        self.whether_equal = whether_equal

    @implement
    def compute(self, predictions, answers):
        return sum(
            int(self.whether_equal(answer=answer, prediction=prediction))
            for prediction, answer in zip(predictions, answers))

    @implement
    def compute_oracle(self, predictions, answers, num_return_sequences):
        # assert len(predictions) == len(answers) * num_return_sequences
        assert len(predictions) == len(answers) * num_return_sequences

        if num_return_sequences is not None and num_return_sequences > 1:
            prediction_groups = tuple(iteration.partition(predictions, num_return_sequences))
        else:
            prediction_groups = predictions

        assert len(prediction_groups) == len(answers)

        num_correct = sum(
            int(any(self.whether_equal(answer=answer, prediction=prediction)
                    for prediction in prediction_group))
            for prediction_group, answer in zip(prediction_groups, answers))

        assert num_correct <= len(answers)

        return num_correct

    @implement
    def get_name(cls):
        return 'num_correct'


class WhetherEqual(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, prediction, answer):
        pass


class Metric(metaclass=ABCMeta):
    @abstractmethod
    def compute(self, score_dict):
        pass

    @classmethod
    @cache
    def get_name(cls):
        return derive_name_from_class(Metric.__name__, cls)

    @property
    def higher_better(self):
        return self.get_higher_better()

    @abstractmethod
    def get_higher_better(self):
        pass

    @staticmethod
    def get_score(score_dict, klass):
        return score_dict[klass.get_name()]


@subclass
class AccuracyMetric(Metric):
    # interface = Interface(Metric)

    @implement
    def compute(self, score_dict):
        return (self.get_score(score_dict, CorrectCounter) /
                self.get_score(score_dict, ExampleCounter))

    @implement
    def get_higher_better(self):
        return True


@subclass
class AccuracyPercentMetric(AccuracyMetric):
    # interface = Interface(AccuracyMetric)

    @implement
    def compute(self, score_dict):
        return ((self.get_score(score_dict, CorrectCounter) * 100) /
                self.get_score(score_dict, ExampleCounter))
