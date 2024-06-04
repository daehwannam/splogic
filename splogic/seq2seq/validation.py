from fractions import Fraction
# import transformers
import warnings
# from itertools import chain
from abc import ABCMeta, abstractmethod
# from functools import cache
from functools import partial
from collections import deque

from accelerate.utils.operations import gather_object

from dhnamlib.pylib import iteration
from dhnamlib.pylib.iteration import pairs2dicts, not_none_valued_pairs, not_none_valued_dict
from dhnamlib.pylib.time import TimeMeasure
# from dhnamlib.pylib.text import camel_to_symbol
# from dhnamlib.pylib.lazy import LazyProxy
# from dhnamlib.pylib.klass import Interface
# from dhnamlib.pylib.klass import subclass, implement, abstractfunction
from dhnamlib.pylib.klass import subclass, implement
from dhnamlib.pylib.torchlib.dnn import unpad_sequence
from dhnamlib.pylib.mllib.learning import get_performance
# from dhnamlib.pylib.torchlib.optimization import get_linear_schedule_with_warmup
from dhnamlib.pylib.structure import XNamespace
from dhnamlib.pylib.hflib.acceleration import alternate_object
from dhnamlib.pylib.decoration import MethodRegister
from dhnamlib.pylib.constant import NO_VALUE

# from configuration import config, coc
# from kqapro.evaluate import denotation_equal

from splogic.utility.tqdm import xtqdm, utqdm
from splogic.utility.acceleration import accelerator
from splogic.base.execution import ExecResult, Executor
from splogic.seq2seq.dynamic_bind import DynamicBinder

from . import learning
from . import decoding
# from .execution import postprocess_prediction


# denotation_equal = LazyProxy(lambda: coc.domain.evaluate.denotation_equal)


class DenotationEqual(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, prediction, answer):
        pass


@subclass
class NaiveDenotationEqual(DenotationEqual):
    @implement
    def __call__(self, prediction, answer):
        return prediction == answer


class ResultCollector:
    method_register = MethodRegister()

    def __init__(self, evaluating, denotation_equal: DenotationEqual, num_return_sequences):
        self._register = self.method_register.instantiate(self)

        self.evaluating = evaluating
        self.denotation_equal = denotation_equal
        self.num_return_sequences = num_return_sequences

        self.batch_queue = deque()

        self.predictions = []
        if evaluating:
            self.answers = []
            self.num_correct = 0

    def collect(self, *, exec_result: ExecResult, answers=None):
        self.batch_queue.append(
            not_none_valued_dict(exec_result=exec_result,
                                 answer=answers))
        self.update()

    def update(self, force=False):
        while len(self.batch_queue) > 0:
            if force or self.batch_queue[0]['exec_result'].is_done():
                batch = self.batch_queue.popleft()
                self._update_from_batch(batch)
            else:
                break

    def _update_from_batch(self, batch):
        predictions = batch['exec_result'].get()
        self.predictions.extend(predictions)

        if self.evaluating:
            assert 'answer' in batch
            answers = batch['answer']
            self.answers.extend(answers)

            self.num_correct += compute_num_correct(
                predictions, answers, self.denotation_equal,
                num_return_sequences=self.num_return_sequences)

    def wait_for_all_batches(self):
        self.update(force=True)

    @method_register('accuracy')
    def get_accuracy(self):
        assert len(self.predictions) == len(self.answers) * self.num_return_sequences
        return (self.num_correct / len(self.answers))

    @method_register('accuracy_percent')
    def get_accuracy_percent(self):
        return self.get_accuracy() * 100

    def get_measure_value(self, measure_name, default=NO_VALUE):
        if len(self.predictions) > 0:
            return self._register.retrieve(measure_name)()
        else:
            assert default is not NO_VALUE
            return default

    def get_overall_performance(self, measure_names, with_extra=False):
        measure_kv_list = []
        measure_cnt = 0

        overall_num_correct = sum(gather_object([self.num_correct]))
        overall_num_answers = sum(gather_object([len(self.answers)]))

        if 'accuracy' in measure_names:
            accuracy_measure_name = 'oracle_accuracy' if self.num_return_sequences > 1 else 'accuracy'
            measure_cnt += 1
            overall_accuracy = overall_num_correct / overall_num_answers
            overall_accuracy_fraction = Fraction(overall_num_correct, overall_num_answers)
            measure_kv_list.append([accuracy_measure_name, overall_accuracy])
            measure_kv_list.append([f'{accuracy_measure_name}_fraction', overall_accuracy_fraction])

        assert len(measure_names) == measure_cnt

        overall_performance = get_performance(measure_kv_list)

        if with_extra:
            # extra_performance = get_performance()
            extra_performance = None
            return overall_performance, extra_performance
        else:
            return overall_performance


class Validator:
    def __init__(
            self,
            compiler,
            context_creator,
            executor: Executor,
            dynamic_binder,
            denotation_equal: DenotationEqual,
            result_collector_cls: ResultCollector,
            extra_analysis_keys=(),
            evaluating_in_progress=True,
    ):
        self.compiler = compiler
        self.context_creator = context_creator
        self.executor = executor
        self.dynamic_binder = dynamic_binder
        self.denotation_equal = denotation_equal
        self.result_collector_cls = result_collector_cls
        self.extra_analysis_keys = extra_analysis_keys
        self.evaluating_in_progress = evaluating_in_progress

    def validate(
            self,
            *,
            grammar,
            model,
            data_loader,
            batch_size,
            num_beams,
            generation_max_length,
            analyzing=True,
            softmax_masking,
            constrained_decoding,
            using_arg_candidate,
            using_distinctive_union_types,
            evaluating,
            using_oracle=False,
            collecting_weaksup_examples=False,
            # strict_postprocessing=False,
            ignoring_parsing_errors=True,
            measure_name='accuracy',
            using_percent_for_progress=True,
    ):
        assert not model.training

        xns = XNamespace()

        num_all_examples = 0

        if using_oracle:
            assert batch_size > 1
            num_return_sequences = num_beams
        else:
            num_return_sequences = 1

        result_collector = self.result_collector_cls(
            evaluating=evaluating,
            denotation_equal=self.denotation_equal,
            num_return_sequences=num_return_sequences)

        if analyzing or collecting_weaksup_examples:
            xns.all_example_ids = []
            xns.all_predicted_token_id_seqs = []

        if analyzing:
            xns.all_dynamic_bindings = []
            xns.all_utterances = []
            xns.all_predicted_last_states = []

            if evaluating:
                xns.all_answer_last_states = []

            extra_analysis_dict = {}

        if collecting_weaksup_examples:
            xns.all_utterance_token_id_seqs = []

        if evaluating and self.evaluating_in_progress:
            tqdm_fn = utqdm
            progress_unit_name = f'oracle_{measure_name}' if using_oracle else measure_name
            default_measure_value = 'none'

            def tqdm_update_fn():
                return result_collector.get_measure_value(
                    measure_name, default=default_measure_value) * \
                    (100 if using_percent_for_progress else 1)

            tqdm_kwargs = dict(
                unit=progress_unit_name,
                update_fn=tqdm_update_fn,
                repr_format='{:5.2f}',
                init_repr=default_measure_value
            )
        else:
            tqdm_fn = xtqdm
            tqdm_kwargs = dict()

        if collecting_weaksup_examples:
            assert using_oracle
            assert evaluating
            assert num_beams > 1
            # assert strict_postprocessing

        all_decoding_time = 0
        tm = TimeMeasure()

        unwrapped_model = accelerator.unwrap_model(model)

        # tqdm_fn = xtqdm  # DEBUG for time measure
        # tqdm_kwargs = dict()    # DEBUG for time measure

        # print('---- Remove debug code ----')
        # debug_batch_idx = -1
        for batch in tqdm_fn(data_loader, **tqdm_kwargs):
            # if debug_batch_idx > 5:
            #     break
            # else:
            #     debug_batch_idx += 1

            dynamic_bindings = self.dynamic_binder.bind_batch(batch, grammar=grammar)

            assert constrained_decoding or not softmax_masking
            if constrained_decoding:
                logits_processor = decoding.get_logits_processor(
                    grammar, batch_size, num_beams, renormalizing=softmax_masking,
                    # utterance_token_ids=batch['utterance_token_ids']
                    dynamic_bindings=dynamic_bindings)
            else:
                logits_processor = None

            tm.check()
            token_id_seqs = decoding.generate_token_id_seqs(
                grammar=grammar,
                model=unwrapped_model,
                utterance_token_ids=batch['utterance_token_ids'].to(unwrapped_model.device),
                max_length=generation_max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                logits_processor=logits_processor,
                # **generation_kwargs
            )
            all_decoding_time += tm.elapse()
            # continue                # DEBUG for time measure

            ignoring_errors = ignoring_parsing_errors or not (
                constrained_decoding and using_arg_candidate and using_distinctive_union_types)
            last_states = decoding.token_id_seqs_to_last_states(
                grammar, token_id_seqs,
                ignoring_parsing_errors=ignoring_errors,
                verifying=False,
                dynamic_bindings=dynamic_bindings,
                num_return_sequences=num_return_sequences
            )
            programs = decoding.last_states_to_programs(
                grammar, self.compiler, last_states, dynamic_bindings,
                num_return_sequences=num_return_sequences,
                tolerant=True, ignoring_compilation_errors=ignoring_errors)

            num_all_examples += batch['utterance_token_ids'].shape[0]
            exec_result = self.executor.execute(
                programs=programs,
                contexts=decoding.repeat_for_multiple_returns(
                    self.context_creator(batch), num_return_sequences))
            # exec_result = executor.execute(programs=programs, contexts=(context,) * len(programs))
            # predictions = learning.programs_to_predictions(context, programs, strict_postprocessing=strict_postprocessing)

            if evaluating:
                assert 'answer' in batch
                answers = batch['answer']
                result_collector.collect(exec_result=exec_result, answers=answers)
            else:
                result_collector.collect(exec_result=exec_result)

            if analyzing or collecting_weaksup_examples:
                xns.all_example_ids.extend(batch['example_id'])
                xns.all_predicted_token_id_seqs.extend(token_id_seqs)

            if analyzing:
                xns.all_dynamic_bindings.extend(dynamic_bindings)

                utterances = grammar.utterance_tokenizer.batch_decode(
                    batch['utterance_token_ids'], skip_special_tokens=True)

                xns.all_utterances.extend(utterances)
                xns.all_predicted_last_states.extend(last_states)

                if evaluating:
                    assert 'labels' in batch
                    answer_last_states = decoding.token_id_seqs_to_last_states(
                        grammar, batch['labels'].tolist(),
                        ignoring_parsing_errors=ignoring_errors,
                        verifying=True,  # config.debug,
                        dynamic_bindings=dynamic_bindings,
                        # utterance_token_id_seqs=(batch['utterance_token_ids'].tolist() if using_arg_candidate else None)
                    )
                    xns.all_answer_last_states.extend(answer_last_states)

                for analysis_key in self.extra_analysis_keys:
                    extra_analysis_dict.setdefault(analysis_key, []).extend(batch[analysis_key])

            if collecting_weaksup_examples:
                xns.all_utterance_token_id_seqs.extend(unpad_sequence(
                    batch['utterance_token_ids'].tolist(), grammar.lf_tokenizer.pad_token_id))

        # coc.logger.info('All decoding time: {} second'.format(all_decoding_time))
        # print('All decoding time: {} second'.format(all_decoding_time))  # DEBUG for time measure
        # import sys; sys.exit(0)      # DEBUG for time measure

        accelerator.wait_for_everyone()

        result_collector.wait_for_all_batches()
        assert len(result_collector.predictions) == num_all_examples * num_return_sequences

        if evaluating:
            assert len(result_collector.predictions) == len(result_collector.answers) * num_return_sequences
            if analyzing:
                assert len(result_collector.answers) == len(xns.all_answer_last_states)

            overall_performance, extra_performance = result_collector.get_overall_performance([measure_name], with_extra=True)

            if collecting_weaksup_examples:
                consistent_action_id_seq_groups = get_consistent_action_id_seq_groups(
                    xns.pop(all_predicted_token_id_seqs=not analyzing),
                    result_collector.predictions,
                    result_collector.answers,
                    self.denotation_equal,
                    num_return_sequences)

                weaksup_examples = tuple(
                    example for example in pairs2dicts(
                        example_id=xns.pop(all_example_ids=not analyzing),
                        utterance_token_ids=xns.pop(all_utterance_token_id_seqs=True),
                        answer=result_collector.answers,
                        action_id_seq_group=consistent_action_id_seq_groups)
                    if len(example['action_id_seq_group']) > 0
                )
                overall_weaksup_examples = sorted(gather_object(weaksup_examples), key=lambda example: example['example_id'])
        else:
            overall_performance = None
            extra_performance = None

        if analyzing:
            analysis = analyze(
                grammar=grammar,
                constrained_decoding=constrained_decoding,
                num_return_sequences=num_return_sequences,
                evaluating=evaluating,
                example_ids=xns.pop(all_example_ids=True),
                dynamic_bindings=xns.pop(all_dynamic_bindings=True),
                utterances=xns.pop(all_utterances=True),
                predicted_last_states=xns.pop(all_predicted_last_states=True),
                answer_last_states=xns.pop(all_answer_last_states=True) if evaluating else None,
                predicted_token_id_seqs=xns.pop(all_predicted_token_id_seqs=True),
                predictions=result_collector.predictions,
                answers=result_collector.answers if evaluating else None,
                denotation_equal=self.denotation_equal,
                **extra_analysis_dict,
            )

            overall_analysis = alternate_object(analysis, batch_size=batch_size)

        if len(xns) > 0:
            raise Exception('There is an existing variable: {}'.format(', '.join(xns)))

        def get_time_info(overall_decoding_time, overall_num_examples):
            average_time = overall_decoding_time / overall_num_examples
            return dict(
                overall_decoding_time=overall_decoding_time,
                average_time=overall_decoding_time / overall_num_examples,
                average_time_millisecond=average_time * 1000
            )

        overall_decoding_time = max(gather_object([all_decoding_time]))
        num_overall_examples = sum(gather_object([num_all_examples]))

        overall_predictions = alternate_object(
            result_collector.predictions,
            batch_size=batch_size * num_return_sequences)

        validation = dict(not_none_valued_pairs(
            performance=overall_performance,
            extra_performance=extra_performance,
            analysis=overall_analysis if analyzing else None,
            weaksup_examples=overall_weaksup_examples if collecting_weaksup_examples else None,
            time_info=get_time_info(overall_decoding_time, num_overall_examples),
            predictions=overall_predictions))

        return validation


def analyze(
        grammar, constrained_decoding, num_return_sequences, evaluating,
        example_ids, dynamic_bindings, utterances, predicted_last_states, answer_last_states,
        predicted_token_id_seqs, predictions, answers, denotation_equal,
        **extra_analysis_dict
):
    def get_action_seq(last_state):
        if last_state is grammar.search_state_cls.INVALID:
            return None
        else:
            if last_state.tree.is_closed_root():
                return list(map(repr, last_state.tree.get_values()))
            else:
                return None

    def get_tree_repr(last_state):
        if last_state is grammar.search_state_cls.INVALID:
            return None
        else:
            return repr(last_state.tree)

    def get_expr_str(dynamic_binding, last_state, expr_key=None):
        if last_state is grammar.search_state_cls.INVALID:
            return None
        else:
            if last_state.tree.is_closed_root():
                # with grammar.dynamic_scope.let(**dynamic_binding):  # debug
                #     return last_state.tree.get_expr_str(expr_key=expr_key)  # debug
                try:
                    with grammar.dynamic_scope.let(**dynamic_binding):
                        return last_state.tree.get_expr_str(expr_key=expr_key)
                except Exception as error:
                    if constrained_decoding:
                        warnings.warn('Error occured during get_expr_str')
                        warnings.warn(repr(error))
                        # raise error
                        return None
                    else:
                        return None
            else:
                return None

    def analyze_program(dynamic_bindings, last_states, token_id_seqs=None, num_return_sequences=1):
        _dynamic_bindings = decoding.repeat_for_multiple_returns(dynamic_bindings, num_return_sequences)
        program_analysis = list(pairs2dicts(not_none_valued_pairs(
            tokens=list(map(grammar.lf_tokenizer.convert_ids_to_tokens, token_id_seqs)) if token_id_seqs is not None else None,
            action_seq=list(map(get_action_seq, last_states)),
            tree=list(map(get_tree_repr, last_states)),
            expr=list(map(get_expr_str, _dynamic_bindings, last_states)),
            # visual_expr=list(map(lambda last_state: get_expr_str(last_state, expr_key='visual'), last_states)),
            visual_expr=list(map(partial(get_expr_str, expr_key='visual'), _dynamic_bindings, last_states)),
        )))
        return program_analysis

    def group_predictions_conditionally(predictions):
        if num_return_sequences > 1:
            prediction_groups = tuple(iteration.partition(predictions, num_return_sequences))
        else:
            prediction_groups = predictions
        return prediction_groups

    if evaluating:
        correct_list = [denotation_equal(prediction=prediction, answer=answer)
                        for prediction, answer in zip(predictions, answers)]
    else:
        correct_list = None

    analysis = list(pairs2dicts(not_none_valued_pairs(
        example_id=example_ids,
        utterance=utterances,
        answer=answers if evaluating else None,
        prediction=group_predictions_conditionally(predictions),
        correct=correct_list,
        predicted_program=group_predictions_conditionally(
            analyze_program(dynamic_bindings, predicted_last_states, predicted_token_id_seqs,
                            num_return_sequences=num_return_sequences)),
        answer_program=(analyze_program(dynamic_bindings, answer_last_states) if evaluating else None),
        **extra_analysis_dict,
    )))

    return analysis


def compute_num_correct(predictions, answers, denotation_equal, num_return_sequences=1):
    assert len(predictions) == len(answers) * num_return_sequences

    num_correct = sum(map(int, compute_correctness(
        predictions, answers, denotation_equal, num_return_sequences=num_return_sequences)))

    assert num_correct <= len(answers)

    return num_correct


def compute_correctness(predictions, answers, denotation_equal, num_return_sequences=1):
    assert len(predictions) == len(answers) * num_return_sequences

    if num_return_sequences > 1:
        correctness_values = compute_oracle_correctness(
            predictions, answers, denotation_equal, num_return_sequences=num_return_sequences)
    else:
        correctness_values = tuple(
            denotation_equal(prediction=prediction, answer=answer)
            for prediction, answer in zip(predictions, answers))

    return correctness_values


def compute_num_oracle_correct(predictions, answers, denotation_equal, num_return_sequences=1):
    num_correct = sum(map(int, compute_oracle_correctness(
        predictions, answers, denotation_equal, num_return_sequences=num_return_sequences)))

    assert num_correct <= len(answers)

    return num_correct


def compute_oracle_correctness(predictions, answers, denotation_equal, num_return_sequences=1):
    # if not (len(predictions) == len(answers) * num_return_sequences):
    #     breakpoint()
    assert len(predictions) == len(answers) * num_return_sequences

    if num_return_sequences is not None and num_return_sequences > 1:
        prediction_groups = tuple(iteration.partition(predictions, num_return_sequences))
    else:
        prediction_groups = predictions

    assert len(prediction_groups) == len(answers)

    correctness_values = tuple(
        any(denotation_equal(prediction=prediction, answer=answer)
            for prediction in prediction_group)
        for prediction_group, answer in zip(prediction_groups, answers))

    return correctness_values


def get_consistent_action_id_seq_groups(action_id_seqs, predictions, answers, denotation_equal, num_return_sequences):
    assert len(action_id_seqs) == len(predictions) == len(answers) * num_return_sequences

    action_id_seq_groups = tuple(iteration.partition(action_id_seqs, num_return_sequences))
    prediction_groups = tuple(iteration.partition(predictions, num_return_sequences))

    assert len(action_id_seq_groups) == len(prediction_groups) == len(answers)

    consistent_action_id_seq_groups = []

    for action_id_seq_group, prediction_group, answer in zip(action_id_seq_groups, prediction_groups, answers):
        consistent_action_id_seq_group = []
        for action_id_seq, prediction in zip(action_id_seq_group, prediction_group):
            if denotation_equal(prediction=prediction, answer=answer):
                consistent_action_id_seq_group.append(action_id_seq)
        consistent_action_id_seq_groups.append(consistent_action_id_seq_group)

    return consistent_action_id_seq_groups
