from fractions import Fraction
# import transformers
import warnings
# from itertools import chain
from abc import ABCMeta, abstractmethod
# from functools import cache

from accelerate.utils.operations import gather_object

from dhnamlib.pylib import iteration
from dhnamlib.pylib.iteration import pairs2dicts, not_none_valued_pairs
from dhnamlib.pylib.time import TimeMeasure
# from dhnamlib.pylib.text import camel_to_symbol
# from dhnamlib.pylib.lazy import LazyProxy
# from dhnamlib.pylib.klass import Interface
# from dhnamlib.pylib.klass import subclass, implement, abstractfunction
from dhnamlib.pylib.torchlib.dnn import unpad_sequence
from dhnamlib.pylib.mllib.learning import get_performance
# from dhnamlib.pylib.torchlib.optimization import get_linear_schedule_with_warmup
from dhnamlib.pylib.structure import XNamespace
from dhnamlib.pylib.hflib.acceleration import alternate_object
from dhnamlib.pylib.decoration import Register

# from configuration import config, coc
# from kqapro.evaluate import denotation_equal

from splogic.utility.tqdm import xtqdm, utqdm
from splogic.utility.acceleration import accelerator

from . import learning
# from .execution import postprocess_prediction


# denotation_equal = LazyProxy(lambda: coc.domain.evaluate.denotation_equal)


class DenotationEqual(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, prediction, answer):
        pass


class PredictionCollector:
    measure_register = Register()

    def __init__(self, evaluating, denotation_equal: DenotationEqual, num_return_sequences):
        self.evaluating = evaluating
        self.denotation_equal = denotation_equal
        self.num_return_sequences = num_return_sequences

        self.predictions = []
        if evaluating:
            self.answers = []
            self.num_correct = 0

    def collect(self, *, predictions, answers=None):
        self.predictions.extend(predictions)
        if self.evaluating:
            self.answers.extend(answers)
            self.num_correct += compute_num_correct(
                predictions, answers, self.denotation_equal,
                num_return_sequences=self.num_return_sequences)

    @measure_register('accuracy')
    def get_accuracy(self):
        assert len(self.predictions) == len(self.answers) * self.num_return_sequences
        return (self.num_correct / len(self.answers))

    @measure_register('accuracy_percent')
    def get_accuracy_percent(self):
        return self.get_accuracy() * 100

    def get_measure_value(self, measure_name):
        return self.measure_register.retrieve(measure_name)()

    def get_overall_performance(self, measure_names):
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
        return overall_performance


def validate(
        *,
        grammar,
        compiler,
        model,
        context,
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
        denotation_equal: DenotationEqual,
        using_oracle=False,
        collecting_weaksup_examples=False,
        strict_postprocessing=False,
        ignoring_parsing_errors=True,
        measure_name='accuracy',
        prediction_collector_cls=PredictionCollector,
):
    assert not model.training

    xns = XNamespace()

    num_all_examples = 0

    if using_oracle:
        assert batch_size > 1
        num_return_sequences = num_beams
    else:
        num_return_sequences = 1

    pred_collector = prediction_collector_cls(
        evaluating=evaluating,
        denotation_equal=denotation_equal,
        num_return_sequences=num_return_sequences)

    if analyzing or collecting_weaksup_examples:
        xns.all_example_ids = []
        xns.all_predicted_token_id_seqs = []

    if analyzing:
        xns.all_utterances = []
        xns.all_predicted_last_states = []

        if evaluating:
            xns.all_answer_last_states = []

    if collecting_weaksup_examples:
        xns.all_utterance_token_id_seqs = []

    if evaluating:
        tqdm_fn = utqdm
        full_measure_name = f'oracle_{measure_name}' if using_oracle else measure_name
        tqdm_kwargs = dict(
            unit=full_measure_name,
            update_fn=pred_collector.get_measure_value(f'{measure_name}_percent'),
            repr_format='{:5.2f}',
            init_repr='none'
        )
    else:
        tqdm_fn = xtqdm
        tqdm_kwargs = dict()

    if collecting_weaksup_examples:
        assert using_oracle
        assert evaluating
        assert num_beams > 1
        assert strict_postprocessing

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

        assert constrained_decoding or not softmax_masking
        if constrained_decoding:
            logits_processor = learning.get_logits_processor(
                grammar, batch_size, num_beams, renormalizing=softmax_masking,
                utterance_token_ids=batch['utterance_token_ids'])
        else:
            logits_processor = None

        tm.check()
        token_id_seqs = learning.generate_token_id_seqs(
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
        last_states = learning.token_id_seqs_to_last_states(
            grammar, token_id_seqs,
            ignoring_parsing_errors=ignoring_errors,
            verifying=False,
            utterance_token_id_seqs=(batch['utterance_token_ids'].tolist() if using_arg_candidate else None),
            num_return_sequences=num_return_sequences
        )
        programs = learning.last_states_to_programs(
            grammar, compiler, last_states, tolerant=True, ignoring_compilation_errors=ignoring_errors)

        num_all_examples += batch['utterance_token_ids'].shape[0]
        predictions = learning.programs_to_predictions(context, programs, strict_postprocessing=strict_postprocessing)

        if evaluating:
            assert 'answer' in batch
            answers = batch['answer']
            pred_collector.collect(predictions=predictions, answers=answers)
        else:
            pred_collector.collect(predictions=predictions)

        if analyzing or collecting_weaksup_examples:
            xns.all_example_ids.extend(batch['example_id'])
            xns.all_predicted_token_id_seqs.extend(token_id_seqs)

        if analyzing:
            utterances = grammar.utterance_tokenizer.batch_decode(
                batch['utterance_token_ids'], skip_special_tokens=True)

            xns.all_utterances.extend(utterances)
            xns.all_predicted_last_states.extend(last_states)

            if evaluating:
                assert 'labels' in batch
                answer_last_states = learning.token_id_seqs_to_last_states(
                    grammar, batch['labels'].tolist(),
                    ignoring_parsing_errors=ignoring_errors,
                    verifying=True,  # config.debug,
                    utterance_token_id_seqs=(batch['utterance_token_ids'].tolist() if using_arg_candidate else None))
                xns.all_answer_last_states.extend(answer_last_states)

        if collecting_weaksup_examples:
            xns.all_utterance_token_id_seqs.extend(unpad_sequence(
                batch['utterance_token_ids'].tolist(), grammar.lf_tokenizer.pad_token_id))

    # coc.logger.info('All decoding time: {} second'.format(all_decoding_time))
    # print('All decoding time: {} second'.format(all_decoding_time))  # DEBUG for time measure
    # import sys; sys.exit(0)      # DEBUG for time measure

    accelerator.wait_for_everyone()

    assert len(pred_collector.predictions) == num_all_examples * num_return_sequences

    if evaluating:
        assert len(pred_collector.predictions) == len(pred_collector.answers) * num_return_sequences
        if analyzing:
            assert len(pred_collector.answers) == len(xns.all_answer_last_states)

        overall_performance = pred_collector.get_overall_performance([measure_name])

        if collecting_weaksup_examples:
            consistent_action_id_seq_groups = get_consistent_action_id_seq_groups(
                xns.pop(all_predicted_token_id_seqs=not analyzing),
                pred_collector.predictions,
                pred_collector.answers,
                denotation_equal,
                num_return_sequences)

            weaksup_examples = tuple(
                example for example in pairs2dicts(
                    example_id=xns.pop(all_example_ids=not analyzing),
                    utterance_token_ids=xns.pop(all_utterance_token_id_seqs=True),
                    answer=pred_collector.answers,
                    action_id_seq_group=consistent_action_id_seq_groups)
                if len(example['action_id_seq_group']) > 0
            )
            overall_weaksup_examples = sorted(gather_object(weaksup_examples), key=lambda example: example['example_id'])
    else:
        overall_performance = None

    if analyzing:
        analysis = analyze(
            grammar=grammar,
            constrained_decoding=constrained_decoding,
            num_return_sequences=num_return_sequences,
            evaluating=evaluating,
            example_ids=xns.pop(all_example_ids=True),
            utterances=xns.pop(all_utterances=True),
            predicted_last_states=xns.pop(all_predicted_last_states=True),
            answer_last_states=xns.pop(all_answer_last_states=True) if evaluating else None,
            predicted_token_id_seqs=xns.pop(all_predicted_token_id_seqs=True),
            predictions=pred_collector.predictions,
            answers=pred_collector.answers if evaluating else None,
            denotation_equal=denotation_equal,
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
        pred_collector.predictions,
        batch_size=batch_size * num_return_sequences)

    validation = dict(not_none_valued_pairs(
        performance=overall_performance,
        analysis=overall_analysis if analyzing else None,
        weaksup_examples=overall_weaksup_examples if collecting_weaksup_examples else None,
        time_info=get_time_info(overall_decoding_time, num_overall_examples),
        predictions=overall_predictions))

    return validation


def analyze(
        grammar, constrained_decoding, num_return_sequences, evaluating,
        example_ids, utterances, predicted_last_states, answer_last_states,
        predicted_token_id_seqs, predictions, answers, denotation_equal
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

    def get_expr_str(last_state, expr_key=None):
        if last_state is grammar.search_state_cls.INVALID:
            return None
        else:
            if last_state.tree.is_closed_root():
                try:
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

    def analyze_program(last_states, token_id_seqs=None):
        program_analysis = list(pairs2dicts(not_none_valued_pairs(
            tokens=list(map(grammar.lf_tokenizer.convert_ids_to_tokens, token_id_seqs)) if token_id_seqs is not None else None,
            action_seq=list(map(get_action_seq, last_states)),
            tree=list(map(get_tree_repr, last_states)),
            expr=list(map(get_expr_str, last_states)),
            visual_expr=list(map(lambda last_state: get_expr_str(last_state, expr_key='visual'), last_states)),
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
            analyze_program(predicted_last_states, predicted_token_id_seqs)),
        answer_program=(analyze_program(answer_last_states) if evaluating else None),
    )))

    return analysis


def compute_num_correct(predictions, answers, denotation_equal, num_return_sequences=1):
    assert len(predictions) == len(answers) * num_return_sequences

    if num_return_sequences > 1:
        num_correct = compute_num_oracle_correct(
            predictions, answers, denotation_equal, num_return_sequences=num_return_sequences)
    else:
        num_correct = sum(
            int(denotation_equal(prediction=prediction, answer=answer))
            for prediction, answer in zip(predictions, answers))

    return num_correct


def compute_num_oracle_correct(predictions, answers, denotation_equal, num_return_sequences=1):
    # if not (len(predictions) == len(answers) * num_return_sequences):
    #     breakpoint()
    assert len(predictions) == len(answers) * num_return_sequences

    if num_return_sequences is not None and num_return_sequences > 1:
        prediction_groups = tuple(iteration.partition(predictions, num_return_sequences))
    else:
        prediction_groups = predictions

    assert len(prediction_groups) == len(answers)

    num_correct = sum(
        int(any(denotation_equal(prediction=prediction, answer=answer)
                for prediction in prediction_group))
        for prediction_group, answer in zip(prediction_groups, answers))

    assert num_correct <= len(answers)

    return num_correct


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
