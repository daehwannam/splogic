
from itertools import chain
# from abc import abstractmethod
from abc import ABCMeta

from dhnamlib.pylib.iteration import all_same, unique
# from dhnamlib.pylib.decoration import construct
# from dhnamlib.pylib.klass import Interface
from dhnamlib.pylib.klass import subclass, implement, override, abstractfunction
from dhnamlib.pylib.function import identity
# from dhnamlib.pylib.typeutil import typecast

import pygtrie


class Trie(metaclass=ABCMeta):
    @abstractfunction
    def generate_candidate_ids(self, id_seq_prefix):
        pass


@subclass
class SequenceTrie(Trie):
    # interface = Interface(Trie)

    def __init__(self, tokenizer=None, end_of_seq=None, ignoring_errors=False):
        self.tokenizer = tokenizer
        self.end_of_seq = end_of_seq
        self.end_of_seq_id = tokenizer.convert_tokens_to_ids(end_of_seq)
        self.ignoring_errors = ignoring_errors
        self.trie = pygtrie.Trie()

    def _normalize_id_seq(self, id_seq):
        if id_seq[-1] != self.end_of_seq_id:
            return tuple(chain(id_seq, [self.end_of_seq_id]))
        else:
            return tuple(id_seq)

    def add_id_seq(self, id_seq):
        self.trie[self._normalize_id_seq(id_seq)] = True

    def add_token_seq(self, token_seq):
        id_seq = self.tokenizer.convert_tokens_to_ids(token_seq)
        self.add_id_seq(id_seq)

    def add_text(self, text):
        tokenized = self.tokenizer(text, add_special_tokens=False)
        self.add_id_seq(tokenized['input_ids'])

    def __contains__(self, key):
        # "key" is a sequence of ids or tokens
        # return self.trie.get(key, False)
        if isinstance(key, (list, tuple)) and isinstance(key[0], str):
            id_seq = self.tokenizer.convert_tokens_to_ids(key)
        elif isinstance(key, str):
            tokenized = self.tokenizer(key, add_special_tokens=False)
            id_seq = tokenized['input_ids']
        else:
            assert isinstance(key[0], int)
            id_seq = key

        return self._normalize_id_seq(id_seq) in self.trie

    @implement
    def generate_candidate_ids(self, id_seq_prefix, ignoring_errors=False):
        # "id_seq_prefix" is a part of an entire sequence of a key
        try:
            prefix_node, path = self.trie._get_node(id_seq_prefix)
        except KeyError as error:
            if self.ignoring_errors or ignoring_errors:
                candidate_ids = ()
            else:
                raise error
        else:
            candidate_ids = (token_id for token_id, node in prefix_node.children.iteritems())
        return candidate_ids

    def candidate_tokens(self, token_seq_prefix):
        # "id_seq_prefix" is a part of an entire sequence of a key
        id_seq_prefix = self.tokenizer.convert_tokens_to_ids(token_seq_prefix)
        candidate_ids = self.generate_candidate_ids(id_seq_prefix)
        return self.tokenizer.convert_ids_to_tokens(candidate_ids)

    def id_seqs(self):
        return iter(self.trie)

    def token_seqs(self):
        def ids_to_tokens(ids):
            return self.tokenizer.convert_ids_to_tokens(ids[:-1])

        return map(ids_to_tokens, self.id_seqs())

    @classmethod
    def merge(cls, token_trie):
        token_trie = tuple(token_trie)

        assert all_same(token_trie.tokenizer for token_trie in token_trie)
        assert all_same(token_trie.end_of_seq for token_trie in token_trie)

        merged_token_trie = cls(token_trie[0].tokenizer, token_trie[0].end_of_seq)

        for token_trie in token_trie:
            merged_token_trie.trie.update(token_trie.trie)

        return merged_token_trie

    __iter__ = token_seqs

    def clone(self):
        trie = SequenceTrie(
            tokenizer=self.tokenizer,
            end_of_seq=self.end_of_seq,
            ignoring_errors=self.ignoring_errors
        )
        trie.trie = pygtrie.Trie(self.trie)
        return trie


@subclass
class DenseSpanTrie(Trie):
    def __init__(self, id_seq, end_of_seq_id):
        self.id_seq = id_seq    # id_seq does not include BOS and EOS. We also assume "add_prefix_space=True".
        self.end_of_seq_id = end_of_seq_id

        id_to_index_set = dict()
        for index, token_id in enumerate(id_seq):
            id_to_index_set.setdefault(token_id, set()).add(index)
        self.id_to_index_set = id_to_index_set

    # def get_candidate_ids(self, id_seq_prefix, allowing_duplicates=False, sorting=True):
    #     candidates = self._candidate_ids(id_seq_prefix)
    #     if not allowing_duplicates:
    #         candidates = set(id_seq_prefix)
    #     if sorting:
    #         candidates = sorted(candidates)
    #     return id_seq_prefix

    # @construct(lambda x: sorted(set(x)))
    # @construct(set)
    @implement
    def generate_candidate_ids(self, id_seq_prefix):
        if len(id_seq_prefix) == 0:
            yield from self.id_seq
        else:
            token_id_iter = iter(id_seq_prefix)
            index_set = self.id_to_index_set.get(next(token_id_iter), set())
            for token_id in token_id_iter:
                if len(index_set) == 0:
                    break
                next_index_set = set()
                for index in index_set:
                    next_index = index + 1
                    if next_index < len(self.id_seq) and self.id_seq[next_index] == token_id:
                        next_index_set.add(next_index)
                index_set = next_index_set
            if len(index_set) > 0:
                yield self.end_of_seq_id
                for index in index_set:
                    next_index = index + 1
                    if next_index < len(self.id_seq):
                        yield self.id_seq[next_index]
            else:
                yield from []


@subclass
class NoTrie(Trie):
    @implement
    def generate_candidate_ids(self, id_seq_prefix):
        return ()


@subclass
class MergedTrie(Trie):
    # interface = Interface(Trie)

    def __init__(self, tries, allowing_duplicates=True):
        self.tries = tries
        self.preprocess = identity if allowing_duplicates else set

    @implement
    def generate_candidate_ids(self, id_seq_prefix):
        candidate_ids_tuple = tuple(
            trie.generate_candidate_ids(id_seq_prefix)
            for trie in self.tries)

        if len(candidate_ids_tuple) == 1:
            all_candidate_ids = unique(candidate_ids_tuple)
        else:
            all_candidate_ids = chain(*candidate_ids_tuple)

        return tuple(self.preprocess(all_candidate_ids))

        # return tuple(self.preprocess(chain(*(
        #     trie.generate_candidate_ids(id_seq_prefix)
        #     for trie in self.tries))))


if __name__ == '__main__':
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained(
        './pretrained/bart-base',
        add_prefix_space=True)
    end_of_seq = '<end-of-seq>'
    tokenizer.add_tokens([end_of_seq], special_tokens=True)
    trie = SequenceTrie(tokenizer, end_of_seq)

    trie.add_text("I'm old.")
    trie.add_text("You have a cat.")

    print(tuple(trie))
    pass
