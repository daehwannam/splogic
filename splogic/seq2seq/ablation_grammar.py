
from itertools import chain
from functools import cache
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
                **kwargs
        ):
            if non_symbolic:
                assert is_symbolic_action is not None
            if using_common_nl_token_seq:
                assert is_nl_token_seq_action is not None
                assert common_nl_token_seq_expr_dict is not None

            assert non_symbolic or using_common_nl_token_seq

            self.non_symbolic = non_symbolic
            self.is_symbolic_action = is_symbolic_action
            self.using_common_nl_token_seq = using_common_nl_token_seq
            self.is_nl_token_seq_action = is_nl_token_seq_action
            self.common_nl_token_seq_expr_dict = common_nl_token_seq_expr_dict


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

    return AblationSeq2SeqGrammar


# Check
#
# [[file:~/work/kqapro/candexpr-sp/.submodules/splogic/splogic/seq2seq/decoding.py::return grammar.search_state_cls.get_last_state(action_seq, verifying=verifying)]]
#
