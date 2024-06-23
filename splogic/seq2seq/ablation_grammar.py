
import warnings

from itertools import chain
from functools import cache
from copy import copy
from dhnamlib.pylib.klass import subclass, override
from dhnamlib.pylib.klass import deprecated
# from dhnamlib.pylib.klass import subclass, override, implement, abstractfunction
from dhnamlib.pylib.iteration import rmap

# from splogic.base.grammar import Grammar
from splogic.base.formalism import Action


def make_ablation_grammar_cls(grammar_cls):
    @subclass
    class AblationSeq2SeqGrammar(grammar_cls):
        def __init__(
                self, *,
                non_symbolic=False, is_symbolic_action=None,
                using_common_nl_token_seq=False, is_nl_token_seq_action=None, common_nl_token_seq_expr_dict=None,
                naive_arg_ordering=False,
                **kwargs
        ):
            if non_symbolic:
                assert is_symbolic_action is not None
            if using_common_nl_token_seq:
                assert is_nl_token_seq_action is not None
                assert common_nl_token_seq_expr_dict is not None

            # assert non_symbolic or using_common_nl_token_seq or naive_arg_ordering
            if not (non_symbolic or using_common_nl_token_seq or naive_arg_ordering):
                warnings.warn('No ablation option is provided.')

            self.non_symbolic = non_symbolic
            self.is_symbolic_action = is_symbolic_action
            self.using_common_nl_token_seq = using_common_nl_token_seq
            self.is_nl_token_seq_action = is_nl_token_seq_action
            self.common_nl_token_seq_expr_dict = common_nl_token_seq_expr_dict
            self.naive_arg_ordering = naive_arg_ordering

            # self._symbol_beg = self.convert_name_to_action('(nl-token Ġ<)')
            # self._symbol_end = self.convert_name_to_action('(nl-token Ġ>)')
            self._symbol_beg_action = Action(
                name='symbol-beg', act_type='none', param_types=[], expr_dict={})
            self._symbol_end_action = Action(
                name='symbol-end', act_type='none', param_types=[], expr_dict={})
            self._common_nl_token_seq_action = Action(
                name='nl-token-seq',
                act_type='none',
                param_types=['part'],
                rest_idx=0,
                expr_dict=common_nl_token_seq_expr_dict)

            # _actions = list(action for action in kwargs['actions'] if not is_symbolic_action(action))
            _actions = list(kwargs['actions'])
            if naive_arg_ordering:
                self._naive_arg_order_action_names = set()
                _actions = list(map(self._apply_naive_arg_ordering, _actions))
            _actions.extend([self._symbol_beg_action, self._symbol_end_action, self._common_nl_token_seq_action])
            _kwargs = dict(kwargs)
            _kwargs['actions'] = _actions

            super().__init__(**_kwargs)

        @cache
        @override
        def get_search_state_cls(grammar):
            base_search_state_cls = super().get_search_state_cls()

            class AblationSeq2SeqSearchState(base_search_state_cls):
                @classmethod
                def get_last_state(cls, action_seq, initial_state=None, verifying=False):
                    _symbolic_action_seq = grammar._convert_to_original_action_seq(action_seq)
                    return base_search_state_cls.get_last_state(_symbolic_action_seq, initial_state=initial_state, verifying=verifying)

            return AblationSeq2SeqSearchState

        def _convert_to_original_action_seq(self, ablation_action_seq):
            action_seq = ablation_action_seq
            if self.non_symbolic:
                action_seq = self._remove_non_symbolic_actions_from_seq(action_seq)
            if self.using_common_nl_token_seq:
                pass

            return action_seq   # original_action_seq

        def _remove_non_symbolic_actions_from_seq(self, ablation_action_seq):
            action_span = []
            action_seq = []
            for action in ablation_action_seq:
                if action.name == self._symbol_beg_action.name:
                    action_span.append(action)
                elif action.name == self._symbol_end_action.name:
                    action_span.append(action)
                    action_seq.append(self._concatenate_actions(action_span))
                    action_span = []
                elif len(action_span) > 0:
                    action_span.append(action)
                else:
                    action_seq.append(action)
            return action_seq

        @deprecated
        def convert_to_ablation_action_seq(self, original_action_seq):
            # currently non_symbolic is only available in this function
            ablation_action_seq = []
            for action in original_action_seq:
                if self.is_symbolic_action(action):
                    ablation_action_seq.extend(self._tokenize_action(action))
                else:
                    ablation_action_seq.append(action)
            return ablation_action_seq

        def convert_to_ablation_action_tree(self, original_action_tree):
            action_tree = original_action_tree
            if self.naive_arg_ordering:
                action_tree = self._add_naive_arg_ordering_to_tree(action_tree)
            if self.non_symbolic:
                action_tree = self._add_non_symbolic_actions_to_tree(action_tree)
            if self.using_common_nl_token_seq:
                action_tree = self._add_common_nl_token_seq_actions_to_tree(action_tree)

            return action_tree

        def _add_non_symbolic_actions_to_tree(self, original_action_tree):
            def convert_action(action):
                if self.is_symbolic_action(action):
                    return self._tokenize_action(action)
                else:
                    return action
            ablation_action_tree = rmap(convert_action, original_action_tree)
            return ablation_action_tree

        def _concatenate_actions(self, actions):
            assert actions[0] is self._symbol_beg_action
            assert actions[-1] is self._symbol_end_action

            token_names = list(action.expr_dict[self.formalism.default_expr_key] for action in actions[1: -1])
            # new_action_name = self.lf_tokenizer.convert_tokens_to_string(token_names).lstrip().replace(' ', '-')
            new_action_name = self._fast_bart_convert_tokens_to_string(token_names).lstrip().replace(' ', '-')

            return self.name_to_action(new_action_name)

        def _tokenize_action(self, action):
            # nl_token_meta_action = self.meta_name_to_meta_action(self.nl_token_meta_name)
            tokenized_names = self.lf_tokenizer.tokenize(action.name.replace('-', ' '))
            return tuple(chain(
                [self._symbol_beg_action],
                (self.id_to_action(self.token_to_id(tokenized_name))
                 for tokenized_name in tokenized_names),
                [self._symbol_end_action]))

        def _fast_bart_convert_tokens_to_string(self, token_names):
            # Use it instead of `lf_tokenizer.convert_tokens_to_string`.
            return ''.join(token_names).replace('Ġ', ' ')

        def _add_common_nl_token_seq_actions_to_tree(self, original_action_tree):
            def convert_action(action):
                if self.is_nl_token_seq_action(action):
                    return self._common_nl_token_seq_action
                else:
                    return action

            ablation_action_tree = rmap(convert_action, original_action_tree)
            return ablation_action_tree

        def _add_naive_arg_ordering_to_tree(self, original_action_tree):

            def recurse(tree):
                if isinstance(tree, (tuple, list)):
                    parent, *children = tree
                    if self._is_naive_arg_order_action(parent):
                        parent._original_arg_indices
                        # _children = [None] * len(children)
                        # for idx, child in zip(parent._original_arg_indices, children):
                        #     _children[idx] = child
                        _children = tuple(children[idx] for idx in parent._original_arg_indices)
                    else:
                        _children = children
                    return (parent,) + tuple(map(recurse, _children))
                else:
                    return tree

            return recurse(original_action_tree)

        # @cache
        # @override
        # def get_name_to_id_dicts(self):
        #     breakpoint()
        #     name_to_id_dicts = list(super().get_name_to_id_dicts())
        #     max_id = max(max(name_to_id_dict.values()) for name_to_id_dict in name_to_id_dicts)
        #     self._symbol_beg_action.id = max_id + 1
        #     self._symbol_end_action.id = max_id + 2
        #     added_name_to_id_dict = dict(
        #         [[self._symbol_beg_action.name, self._symbol_beg_action.id],
        #          [self._symbol_end_action.name, self._symbol_end_action.id]])
        #     name_to_id_dicts.append(added_name_to_id_dict)
        #     return name_to_id_dicts

        def _apply_naive_arg_ordering(self, action):

            default_expr_key = 'default'
            visual_expr_key = 'visual'

            def get_arg_indices(expr_template):
                return tuple(
                    int(match_obj.group(1))
                    for match_obj in Action._place_holder_regex.finditer(expr_template))

            default_expr_template = action.expr_dict[default_expr_key]

            if callable(default_expr_template):
                return action
            else:
                default_expr_arg_indices = get_arg_indices(default_expr_template)
                sorted_default_expr_arg_indices = tuple(sorted(default_expr_arg_indices))
                if default_expr_arg_indices == sorted_default_expr_arg_indices:
                    return action
                else:
                    new_action = copy(action)
                    new_action._original_expr_dict = action.expr_dict
                    new_action._original_arg_indices = default_expr_arg_indices

                    new_param_types = tuple(
                        action.param_types[arg_index]
                        for arg_index in default_expr_arg_indices)
                    new_action.param_types = new_param_types

                    def change_expr_template(template):
                        # expr_arg_indices = get_arg_indices(template)
                        redirected_indices = [None] * len(default_expr_arg_indices)
                        for new_idx, expr_arg_idx in enumerate(default_expr_arg_indices):
                            redirected_indices[expr_arg_idx] = new_idx
                        return template.format(*(('{' + str(new_idx) + '}') for new_idx in redirected_indices))

                    new_expr_dict = {}
                    new_expr_dict[default_expr_key] = change_expr_template(action.expr_dict[default_expr_key])
                    # print(new_expr_dict[default_expr_key])

                    visual_expr_template = action.expr_dict[visual_expr_key]
                    if (visual_expr_template is not None) and (not callable(visual_expr_template)):
                        # visual_expr_arg_indices = get_arg_indices(visual_expr_template)
                        # assert default_expr_arg_indices == visual_expr_arg_indices
                        new_expr_dict[visual_expr_key] = change_expr_template(action.expr_dict[visual_expr_key])
                    new_action.expr_dict = new_expr_dict
                    new_action.expr_pieces_dict = Action.get_expr_pieces_dict(new_expr_dict)

                    self._naive_arg_order_action_names.add(new_action.name)
                    return new_action

        def _is_naive_arg_order_action(self, action):
            return action.name in self._naive_arg_order_action_names

# TODO
# - def is_arg_reordered_action_name
# - def _convert_to_naive_arg_order_tree

    return AblationSeq2SeqGrammar


# Check
#
# [[file:~/work/kqapro/candexpr-sp/.submodules/splogic/splogic/seq2seq/decoding.py::return grammar.search_state_cls.get_last_state(action_seq, verifying=verifying)]]
#
