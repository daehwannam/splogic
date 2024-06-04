
from typing import List, Tuple, Callable
import math

import torch
import transformers

from splogic.base.formalism import InvalidCandidateActionError
from splogic.base.execution import INVALID_PROGRAM

from dhnamlib.pylib.hflib.transforming import logit_rescaling
from dhnamlib.pylib import iteration
from dhnamlib.pylib.exception import NotFoundError
from dhnamlib.pylib.data_structure import FIFODict
from dhnamlib.pylib.torchlib.dnn import unpad_sequence

# from .execution import postprocess_prediction, invalid_program, get_counting_context


def token_id_seq_to_action_seq(grammar, token_id_seq):
    eos_token_id_idx = iteration.index(token_id_seq, grammar.lf_tokenizer.eos_token_id, reverse=True)
    assert token_id_seq[0] == grammar.lf_tokenizer.bos_token_id
    action_id_seq = token_id_seq[1: eos_token_id_idx]  # index 0 has bos_token_id, which should be skipped
    action_seq = tuple(map(grammar.id_to_action, action_id_seq))
    return action_seq


def token_id_seq_to_last_state(grammar, token_id_seq, ignoring_parsing_errors=False,
                               verifying=False, dynamic_binding={}):
    try:
        action_seq = token_id_seq_to_action_seq(grammar, token_id_seq)

        def get_last_state():
            return grammar.search_state_cls.get_last_state(action_seq, verifying=verifying)

        with grammar.dynamic_scope.let(**dynamic_binding):
            last_state = get_last_state()

        return last_state
    except NotFoundError:
        return grammar.search_state_cls.INVALID
    except Exception as error:
        if ignoring_parsing_errors:
            return grammar.search_state_cls.INVALID
        else:
            raise error


def utterances_to_ids(grammar, utterances):
    encoded_utterances = grammar.utterance_tokenizer(utterances)
    utterance_token_ids = encoded_utterances['input_ids']
    return utterance_token_ids


# @construct(tuple)
def generate_token_id_seqs(
        grammar, model, utterance_token_ids, max_length, num_beams,
        num_return_sequences=None,
        prefix_allowed_tokens_fn=None, logits_processor=transformers.LogitsProcessorList()
        # , **kwargs
):
    """
    It returns a list of token id sequences.
    An output token id sequence starts with `bos_token_id` (the ID of '<s>') and ends with `eos_token_id` (the ID of '</s>').
    To produce the output token sequence, `decoder_start_token_id` and `pad_token_id` are removed.

    The size of the output list depends on `batched_output` from `model.generate`, whose shape is
    (1) (batch_size, seq_len) or
    (2) (batch_size * num_return_sequences, seq_len) if `num_return_sequences` > 1 .

    If num_return_sequences > 1, the output list can be further split as
    a more nested list with a shape (batch_size, num_return_sequences, seq_len)
    where seq_len is different for each token id sequence.
    For example,
    >>> token_id_seq_groups = tuple(iteration.partition(token_id_seqs, num_return_sequences))  # doctest: +SKIP
    >>> assert len(token_id_seq_groups) == batch_size  # doctest: +SKIP
    >>> assert len(token_id_seq_groups[0]) == num_return_sequences  # doctest: +SKIP
    """

    if num_return_sequences is not None:
        assert 1 <= num_return_sequences <= num_beams

    if logits_processor is None:
        logits_processor = transformers.LogitsProcessorList()

    # breakpoint()
    batched_output = model.generate(
        input_ids=utterance_token_ids,
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor
        # **kwargs
    )

    batched_token_ids = batched_output[:, 1:]  # removing `decoder_start_token_id`

    token_id_seqs = []
    for token_id_seq in batched_token_ids.tolist():
        # try:
        #     eos_token_id_idx = iteration.index(token_id_seq, grammar.lf_tokenizer.eos_token_id, reverse=True)
        #     token_id_seq_without_padding = token_id_seq[:eos_token_id_idx + 1]
        # except NotFoundError:
        #     token_id_seq_without_padding = token_id_seq
        token_id_seq_without_padding = unpad_sequence(token_id_seq, grammar.lf_tokenizer.pad_token_id)
        token_id_seqs.append(token_id_seq_without_padding)

    return token_id_seqs


def repeat_for_multiple_returns(coll, num_return_sequences):
    if num_return_sequences > 1:
        _coll = tuple(iteration.repeat_in_order(
            coll, num_return_sequences))
    else:
        _coll = coll
    return _coll


def token_id_seqs_to_last_states(
        grammar, token_id_seqs, *, ignoring_parsing_errors=False, verifying=False,
        dynamic_bindings=None,
        # utterance_token_id_seqs=None,
        num_return_sequences=1
):
    # if dynamic_bindings is None:
    #     _dynamic_bindings = [{}] * len(token_id_seqs)
    # else:
    #     _dynamic_bindings = repeat_for_multiple_returns(dynamic_bindings, num_return_sequences)
    _dynamic_bindings = repeat_for_multiple_returns(dynamic_bindings, num_return_sequences)

    assert len(token_id_seqs) == len(_dynamic_bindings)

    predicted_last_states = tuple(
        token_id_seq_to_last_state(
            grammar, token_id_seq, ignoring_parsing_errors=ignoring_parsing_errors,
            verifying=verifying, dynamic_binding=dynamic_binding)
        for token_id_seq, dynamic_binding in zip(token_id_seqs, _dynamic_bindings))

    return predicted_last_states


def last_states_to_programs(grammar, compiler, last_states, dynamic_bindings, num_return_sequences=1,
                            tolerant=False, ignoring_compilation_errors=False):
    def state_to_program(state, dynamic_binding):
        if state is grammar.search_state_cls.INVALID:
            return INVALID_PROGRAM
        else:
            if state.tree.is_closed_root():
                try:
                    with grammar.dynamic_scope.let(**dynamic_binding):
                        return compiler.compile_tree(state.tree, tolerant=tolerant)
                except Exception as error:
                    if ignoring_compilation_errors:
                        return INVALID_PROGRAM
                    else:
                        raise error
            else:
                # when generation process reaches the max_length
                return INVALID_PROGRAM

    _dynamic_bindings = repeat_for_multiple_returns(dynamic_bindings, num_return_sequences)
    assert len(last_states) == len(_dynamic_bindings)

    programs = tuple(state_to_program(last_state, dynamic_binding)
                     for last_state, dynamic_binding in zip(last_states, _dynamic_bindings))
    return programs


# # @config
# def programs_to_predictions(context, programs, max_num_program_iterations=config.ph, strict_postprocessing=False):
#     predictions = tuple(
#         postprocess_prediction(
#             program(get_counting_context(
#                 context, max_num_iterations=max_num_program_iterations)),
#             strict=strict_postprocessing)
#         for program in programs)

#     return predictions


class SequencePrefixProcessor:
    def __init__(self, grammar, batch_size, num_beams, dynamic_bindings, additional_mask_cache: dict = None):
        self.grammar = grammar
        self.dynamic_bindings = dynamic_bindings

        #  Multiplying "2" is for caching both previous states and the next sates
        self.num_beams = num_beams
        self.cache_size = batch_size * num_beams * 2
        self.state_fifo_dict = FIFODict(self.cache_size)

        self.DECODER_START_TOKEN_ID = grammar.model_config.decoder_start_token_id
        self.BOS_TOKEN_ID = grammar.lf_tokenizer.bos_token_id
        self.EOS_TOKEN_ID = grammar.lf_tokenizer.eos_token_id
        self.PAD_TOKEN_ID = grammar.lf_tokenizer.pad_token_id

        self.vocab_size = len(grammar.lf_tokenizer)
        self.additional_mask_cache = additional_mask_cache

    def action_id_seq_to_state(self, action_id_seq):
        assert isinstance(action_id_seq, tuple)

        if action_id_seq in self.state_fifo_dict:
            return self.state_fifo_dict[action_id_seq]
        else:
            if len(action_id_seq) == 0:
                curr_state = self.grammar.search_state_cls.create()
            elif action_id_seq[:-1] in self.state_fifo_dict:
                prev_state = self.state_fifo_dict[action_id_seq[:-1]]
                if prev_state in [self.grammar.search_state_cls.END, self.grammar.search_state_cls.INVALID]:
                    curr_state = self.grammar.search_state_cls.INVALID
                else:
                    next_action_id_seq = action_id_seq[-1:]  # a list with only the last element
                    try:
                        action_seq = tuple(map(self.grammar.id_to_action, next_action_id_seq))
                    except NotFoundError:
                        curr_state = self.grammar.search_state_cls.INVALID
                    else:
                        if prev_state.tree.is_closed_root():
                            # This block is entered when using beam search,
                            # which examines all items in a beam,
                            # whether the item has a valid or invalid state.
                            assert self.num_beams > 1
                            curr_state = self.grammar.search_state_cls.INVALID
                        else:
                            try:
                                curr_state = self.grammar.search_state_cls.get_last_state(
                                    action_seq, initial_state=prev_state, verifying=True)
                            except InvalidCandidateActionError:
                                curr_state = self.grammar.search_state_cls.INVALID
            else:
                # breakpoint()
                # # curr_state = self.grammar.search_state_cls.INVALID

                # # When this block is entered?
                # #
                # # `action_id_seq_to_state` in called in `prefix_allowed_and_ids_pair_fn`.
                # # However, when last_token_id in [self.PAD_TOKEN_ID, self.EOS_TOKEN_ID],
                # # `action_id_seq_to_state` is not called, then the last states are not saved.
                # #
                # # e.g. action_id_seq == (50305, 50282, 443, 50309, 50280, 50286, 726, 459, 3494, 219, 413, 50309, 1, 7)
                # # where 1 is the id of self.PAD_TOKEN_ID

                raise Exception(
                    'The `cache_size` is not enough. '
                    'Check `batch_size` is synchronized with that of a DataLoader object')

            self.state_fifo_dict[action_id_seq] = curr_state
            return curr_state

    def prefix_allowed_and_ids_pair_fn(self, batch_id: int, prefix_token_id_seq: torch.Tensor) -> List[int]:
        _prefix_token_id_seq = prefix_token_id_seq.tolist()

        # # Start of DEBUG
        # from .kopl_transfer import token_to_action_name
        # test_token_seq = ["<s>", "<count>", "<union>", "<filter-concept>", "<keyword-concept>", "Ġcounty", "Ġof", "ĠPennsylvania", "<reduce>", "<filter-number>", "<keyword-attribute-number>", "Ġpopulation", "<reduce>", "<constant-number>", "<constant-quantity>", "Ġ7", "800", "<reduce>", "<constant-unit>", "<reduce>", "<op-gt>", "<all-entities>", "<filter-concept>", "<keyword-concept>", "Ġcounty", "Ġof", "ĠPennsylvania", "<reduce>", "<filter-number>", "<keyword-attribute-number>", "Ġpopulation", "<reduce>", "<constant-number>"]
        # test_token_id_seq = [self.DECODER_START_TOKEN_ID, self.BOS_TOKEN_ID] + list(
        #     self.grammar.name_to_id(token_to_action_name(token, self.grammar.non_nl_tokens)) for token in test_token_seq[1:])
        # if test_token_id_seq == _prefix_token_id_seq:
        #     # breakpoint()
        #     pass
        # if test_token_id_seq == _prefix_token_id_seq[:-1]:
        #     if _prefix_token_id_seq[-1] == 50268:
        #         breakpoint()
        #         print(50268)
        #     if _prefix_token_id_seq[-1] == 3:
        #         breakpoint()
        #         print(3)
        # # End of DEBUG

        if len(_prefix_token_id_seq) == 1:
            # when `_prefix_token_id_seq` has only `self.DECODER_START_TOKEN_ID`
            return True, [self.BOS_TOKEN_ID]
        else:
            decoder_start_token_id, bos_token_id, *action_id_seq = _prefix_token_id_seq
            assert decoder_start_token_id == self.DECODER_START_TOKEN_ID
            assert bos_token_id == self.BOS_TOKEN_ID

            last_token_id = _prefix_token_id_seq[-1]
            if last_token_id in [self.PAD_TOKEN_ID, self.EOS_TOKEN_ID]:
                self.state_fifo_dict[tuple(action_id_seq)] = self.grammar.search_state_cls.END
                return True, [self.PAD_TOKEN_ID]
            else:
                # decoder_start_token_id, bos_token_id, *action_id_seq = _prefix_token_id_seq
                # assert decoder_start_token_id == self.DECODER_START_TOKEN_ID
                # assert bos_token_id == self.BOS_TOKEN_ID

                with self.grammar.dynamic_scope.let(**self.dynamic_bindings[batch_id]):
                    curr_state = self.action_id_seq_to_state(tuple(action_id_seq))

                if curr_state is self.grammar.search_state_cls.INVALID:
                    return True, []
                elif curr_state.tree.is_closed_root():
                    return True, [self.EOS_TOKEN_ID]
                else:
                    with self.grammar.dynamic_scope.let(**self.dynamic_bindings[batch_id]):
                        return curr_state.get_allowed_and_ids_pairs()


def get_logits_processor(grammar, batch_size, num_beams, renormalizing, dynamic_bindings):
    '''logits processor for constrained decoding'''
    sequence_prefix_processor = SequencePrefixProcessor(grammar, batch_size, num_beams, dynamic_bindings)
    prefix_allowed_and_ids_pair_fn = sequence_prefix_processor.prefix_allowed_and_ids_pair_fn
    fast_prefix_constrained_logits_processor = FastPrefixConstrainedLogitsProcessor(
        prefix_allowed_and_ids_pair_fn, num_beams=num_beams)
    if renormalizing:
        fast_prefix_constrained_logits_processor = logit_rescaling(
            fast_prefix_constrained_logits_processor, postprocessing_nan=(num_beams > 1))
    logits_processor = transformers.LogitsProcessorList([fast_prefix_constrained_logits_processor])
    return logits_processor


class FastPrefixConstrainedLogitsProcessor(transformers.LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. This class is modified from `transformers.PrefixConstrainedLogitsProcessor`.

    Args: prefix_allowed_and_ids_pair_fn: (`Callable[[int, torch.Tensor], Tuple[bool, List[int]]]`): This
        function constraints the beam search to allowed tokens only at each step. This function takes 2
        arguments `inputs_ids` and the batch ID `batch_id`. It has to return a bool object and a token
        ids. When the bool object is True, the token ids should be allowed for the next generation step. When
        the bool object is False, the token ids should not be allowed for the next generation step. The bool
        object and toke ids are created conditioned on the previously generated tokens `inputs_ids` and the
        batch ID `batch_id`.

    """

    def __init__(self, prefix_allowed_and_ids_pair_fn: Callable[[int, torch.Tensor], Tuple[bool, List[int]]], num_beams: int):
        self._prefix_allowed_and_ids_pair_fn = prefix_allowed_and_ids_pair_fn
        self._num_beams = num_beams

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                allowed, token_ids = self._prefix_allowed_and_ids_pair_fn(batch_id, sent)
                if allowed:
                    mask[batch_id * self._num_beams + beam_id, token_ids] = 0
                else:
                    mask[batch_id * self._num_beams + beam_id, :] = 0
                    mask[batch_id * self._num_beams + beam_id, token_ids] = -math.inf

        return scores + mask
