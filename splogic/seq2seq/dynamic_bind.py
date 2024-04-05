
from abc import ABCMeta, abstractmethod

from transformers.models.bart.tokenization_bart import BartTokenizer

from splogic.utility.trie import DenseSpanTrie

from dhnamlib.pylib import iteration
from dhnamlib.pylib.klass import subclass, implement


def _convert_utterance_token_id_seq_to_span_trie(grammar, utterance_token_id_seq):
    eos_token_id_idx = iteration.index(utterance_token_id_seq, grammar.utterance_tokenizer.eos_token_id, reverse=True)
    assert utterance_token_id_seq[0] == grammar.utterance_tokenizer.bos_token_id
    trimmed_utterance_token_id_seq = utterance_token_id_seq[1: eos_token_id_idx]
    # first_utterance_token = grammar.utterance_tokenizer.convert_ids_to_tokens(trimmed_utterance_token_id_seq[0])
    # if not first_utterance_token.startswith('Ġ'):
    #     trimmed_utterance_token_id_seq[0] = grammar.utterance_tokenizer.convert_tokens_to_ids('Ġ' + first_utterance_token)
    end_of_seq_id = grammar.reduce_action_id
    utterance_span_trie = DenseSpanTrie(trimmed_utterance_token_id_seq, end_of_seq_id)
    return utterance_span_trie


def _preprocess_bart_utterance_token_id_seq(grammar, utterance_token_id_seq):
    assert utterance_token_id_seq[0] == grammar.utterance_tokenizer.bos_token_id
    _utterance_token_id_seq = list(utterance_token_id_seq)
    first_utterance_token = grammar.utterance_tokenizer.convert_ids_to_tokens(_utterance_token_id_seq[1])
    if not first_utterance_token.startswith('Ġ'):
        _utterance_token_id_seq[1] = grammar.utterance_tokenizer.convert_tokens_to_ids(
            'Ġ' + first_utterance_token)

    return _utterance_token_id_seq


def utterance_token_id_seq_to_span_trie(grammar, utterance_token_id_seq):
    if isinstance(grammar.utterance_tokenizer, BartTokenizer):
        utterance_token_id_seq = _preprocess_bart_utterance_token_id_seq(grammar, utterance_token_id_seq)
    return _convert_utterance_token_id_seq_to_span_trie(grammar, utterance_token_id_seq)


class DynamicBinder(metaclass=ABCMeta):
    @abstractmethod
    def bind(self, example):
        pass

    @abstractmethod
    def bind_batch(self, grammar, batched_example):
        pass


@subclass
class NoDynamicBinder(DynamicBinder):
    @implement
    def bind(self, grammar, example):
        binding = {}
        return binding

    @implement
    def bind_batch(self, grammar, batched_example):
        batch_size = (len(batched_example['utterance_token_ids']) if 'utterance_token_ids' in batched_example else
                      len(batched_example[next(iter(batched_example))]))
        bindings = ({},) * batch_size
        return bindings


@subclass
class UtteranceSpanTrieDynamicBinder(DynamicBinder):
    @implement
    def bind(self, grammar, example):
        binding = dict(utterance_span_trie=utterance_token_id_seq_to_span_trie(grammar, example['utterance_token_ids']))
        return binding

    @implement
    def bind_batch(self, grammar, batched_example):
        bindings = tuple(
            dict(utterance_span_trie=utterance_token_id_seq_to_span_trie(grammar, utterance_token_id_seq))
            for utterance_token_id_seq in batched_example['utterance_token_ids'].tolist()
        )
        return bindings
