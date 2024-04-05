
import copy
from itertools import chain
from functools import cache

from dhnamlib.pylib.decoration import construct, variable, deprecated
# from dhnamlib.pylib.klass import Interface, abstractfunction
from dhnamlib.pylib.klass import subclass, implement, abstractfunction
from dhnamlib.pylib.context import block
from dhnamlib.pylib.iteration import distinct_values
from dhnamlib.pylib.hflib.transforming import iter_id_token_pairs, join_tokens
from dhnamlib.pylib.torchlib.dnn import candidate_ids_to_mask

from splogic.base.grammar import Grammar, SymbolicRegister
from splogic.base.formalism import make_program_tree_cls, make_search_state_cls
from splogic.utility.trie import Trie

from . import filemng
from .transfer import TokenProcessing, ActionNameStyle, StrictTypeProcessing


_DEFAULT_NL_TOKEN_META_NAME = 'nl-token'
_DEFAULT_NL_TOKEN_META_ARG_NAME = 'token'


@subclass
class Seq2SeqGrammar(Grammar):
    # interface = Interface(Grammar)
    general_register = SymbolicRegister()

    # @config
    def __init__(
            self, *,
            #
            # Parameters of Grammar, which is the super-class
            #
            formalism, super_types_dict, actions, start_action, meta_actions, register,
            is_non_conceptual_type=None, use_reduce=True,
            inferencing_subtypes=True,  # config.ph(True),
            #
            # Parameters that are newly added
            #
            dynamic_scope,
            #
            using_distinctive_union_types=True,  # config.ph(True),
            #
            token_processing: TokenProcessing,
            action_name_style: ActionNameStyle,
            strict_type_processing: StrictTypeProcessing,
            #
            pretrained_model_name_or_path,  # config.ph
            #
            nl_token_meta_name=_DEFAULT_NL_TOKEN_META_NAME,
            nl_token_meta_arg_name=_DEFAULT_NL_TOKEN_META_ARG_NAME,
    ):

        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        if not inferencing_subtypes:
            super_to_sub_actions = strict_type_processing.iter_super_to_sub_actions(super_types_dict, is_non_conceptual_type)
            actions = tuple(chain(actions, super_to_sub_actions))

        self.action_name_style = action_name_style

        super().__init__(formalism=formalism, super_types_dict=super_types_dict, actions=actions, start_action=start_action,
                         meta_actions=meta_actions, register=register, is_non_conceptual_type=is_non_conceptual_type,
                         use_reduce=use_reduce, inferencing_subtypes=inferencing_subtypes)

        self.token_processing = token_processing
        self.strict_type_processing = strict_type_processing
        self.nl_token_meta_name = nl_token_meta_name
        self.nl_token_meta_arg_name = nl_token_meta_arg_name

        self.dynamic_scope = dynamic_scope
        self.model_config = filemng.load_model_config(pretrained_model_name_or_path)
        self.initialize_from_base_actions()
        self.register_all()
        self.add_actions(token_processing.iter_nl_token_actions(
            self.meta_name_to_meta_action(nl_token_meta_name),
            self.lf_tokenizer,
            using_distinctive_union_types=using_distinctive_union_types))

    @cache
    def initialize_from_base_actions(self):
        self.non_nl_tokens = set(distinct_values(
            self.action_name_style.action_name_to_special_token(action.name)
            for action in self.base_actions))

        # logical form tokenizer
        self.lf_tokenizer = filemng.load_tokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            add_prefix_space=True,
            non_nl_tokens=self.non_nl_tokens)

        # utterance tokenizer
        with block:
            self.utterance_tokenizer = copy.copy(self.lf_tokenizer)
            self.utterance_tokenizer.add_prefix_space = False

    @cache
    @implement
    def get_name_to_id_dicts(self):
        self.initialize_from_base_actions()

        @variable
        @construct(dict)
        def name_to_id_dict():
            for token_id, token in iter_id_token_pairs(self.lf_tokenizer):
                action_name = self.action_name_style.token_to_action_name(token, special_tokens=self.non_nl_tokens)
                yield action_name, token_id

        return [name_to_id_dict]

    @cache
    def _get_token_to_id_dict(self):
        self.initialize_from_base_actions()
        return dict(map(reversed, iter_id_token_pairs(self.lf_tokenizer)))

    def token_to_id(self, token):
        return self._get_token_to_id_dict()[token]

    @property
    @cache
    def reduce_token(self):
        return self.action_name_style.action_name_to_special_token(self.reduce_action.name)

    @cache
    @implement
    def get_reduce_action_id(self):
        return self.lf_tokenizer.convert_tokens_to_ids(self.reduce_token)

    @cache
    @implement
    def get_program_tree_cls(self):
        return make_program_tree_cls(self.formalism, name='Seq2SeqProgramTree')

    # @config
    @cache
    @implement
    def get_search_state_cls(
            self,
            using_arg_candidate=True,  # config.ph,
            using_arg_filter=False,  # config.ph
    ):
        @deprecated
        def ids_to_mask_fn(action_ids):
            return candidate_ids_to_mask(action_ids, len(self.lf_tokenizer))

        return make_search_state_cls(
            grammar=self,
            name='Seq2SeqSearchState',
            using_arg_candidate=using_arg_candidate,
            using_arg_filter=using_arg_filter,
            ids_to_mask_fn=ids_to_mask_fn)

    @abstractfunction
    def get_compiler_cls(self):
        pass

    @implement
    def iter_all_token_ids(self):
        return range(len(self.lf_tokenizer))

    # def let_dynamic_trie(self, dynamic_trie, using_spans_as_entities=None):
    #     if using_spans_as_entities is None:
    #         using_spans_as_entities = self.using_spans_as_entities
    #     return self.dynamic_scope.let(dynamic_trie=dynamic_trie,
    #                                   using_spans_as_entities=using_spans_as_entities)

    # def dynamic_let(self, pairs=(), **kwargs):
    #     return self.dynamic_scope.let(pairs, **kwargs)

    def register_all(self):
        self.register_base(self.register)
        self.register_specific(self.register)

    def register_base(self, register):
        register.update(self.general_register.instantiate(self))

    @abstractfunction
    def register_specific(self, register):
        '''
        Register domain-specific items
        '''
        pass

    @general_register('(name nl-token)')
    def get_nl_token_name(self, token):
        return self.action_name_style.nl_token_to_action_name(token)

    @general_register('(function join-nl-tokens)')
    def join_nl_tokens(self, tokens):
        return join_tokens(self.lf_tokenizer, tokens, skip_special_tokens=True).lstrip()

    @abstractfunction
    def fast_join_nl_tokens(self, tokens):
        pass

    @general_register('(function concat-nl-tokens)')
    def concat_nl_tokens(self, *tokens):
        return self.join_nl_tokens(tokens)

    def make_arg_filter(self, is_valid_prefix, is_valid_expr):
        def arg_filter(tree, action_ids):
            opened_tree, children = tree.get_opened_tree_children()
            for action_id in action_ids:
                action = self.id_to_action(action_id)
                token_seq = [child.value.get_meta_arg(self.nl_token_meta_arg_name) for child in children]
                if action is self.reduce_action:
                    pass
                else:
                    token_seq.append(action.get_meta_arg(self.nl_token_meta_arg_name))
                joined_tokens = self.fast_join_nl_tokens(token_seq)
                if action == self.reduce_action:
                    if is_valid_expr(joined_tokens):
                        yield action_id
                else:
                    if is_valid_prefix(joined_tokens):
                        yield action_id
        return arg_filter

    def make_trie_arg_candidate(self, trie: Trie):
        def arg_candidate(tree):
            opened_tree, children = tree.get_opened_tree_children()
            id_seq_prefix = tuple(self.token_to_id(child.value.get_meta_arg(self.nl_token_meta_arg_name))
                                  for child in children)
            return tuple(trie.generate_candidate_ids(id_seq_prefix))
        return arg_candidate


if __name__ == '__main__':
    pass
