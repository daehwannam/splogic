
"Transferring logical forms to actions"

from itertools import chain
from collections import deque

from dhnamlib.pylib.hflib.transforming import iter_default_non_special_tokens
from dhnamlib.pylib.klass import abstractfunction
from dhnamlib.pylib.iteration import flatten


class TokenProcessing:
    # def __init__(self, nl_token_meta_name):
    #     # e.g. nl_token_meta_name == 'nl-token'
    #     self.nl_token_meta_name = nl_token_meta_name

    def iter_nl_token_actions(
            self, nl_token_meta_action, lf_tokenizer,
            using_distinctive_union_types=True
    ):
        "Convert tokens to nl-token actions"

        if using_distinctive_union_types:
            _get_token_act_type = self.get_token_act_type
        else:
            _get_token_act_type = self.get_non_distinctive_nl_token_act_type

        def iter_token_value_act_type_pairs():
            all_non_special_token_values = tuple(iter_default_non_special_tokens(lf_tokenizer))
            all_act_types = map(_get_token_act_type, all_non_special_token_values)

            return zip(all_non_special_token_values, all_act_types)

        def iter_token_actions():
            for token_value, act_type in iter_token_value_act_type_pairs():
                yield nl_token_meta_action(meta_kwargs=dict(token=token_value),
                                           act_type=act_type)

        return iter_token_actions()

    @abstractfunction
    def get_token_act_type(self, token_value):
        pass

    @abstractfunction
    def get_non_distinctive_nl_token_act_type(self, token_value):
        """
        The act-type for nl-token action when using_distinctive_union_types=False.
        """

        pass

    @abstractfunction
    def labeled_logical_form_to_action_seq(labeled_logical_form, *args, **kwargs):
        pass

    @abstractfunction
    def labeled_logical_form_to_action_tree(grammar, context, labeled_logical_form):
        pass


class ActionNameStyle:
    # Conversion bewteen tokens and action names

    def __init__(self, nl_token_meta_name):
        # e.g. nl_token_meta_name == 'nl-token'
        self.nl_token_meta_name = nl_token_meta_name
        self._delimiter_after_prefix = ' '

    @staticmethod
    def action_name_to_special_token(action_name):
        return f'<{action_name}>'

    @staticmethod
    def special_token_to_action_name(special_token):
        assert special_token[0] == '<'
        assert special_token[-1] == '>'
        return special_token[1:-1]

    def _add_prefix(self, text, prefix):
        return f'{prefix}{self._delimiter_after_prefix}{text}'

    def _remove_prefix(self, text_with_prefix, prefix):
        return text_with_prefix[len(prefix) + len(self._delimiter_after_prefix):]

    @staticmethod
    def _add_parentheses(text):
        return f'({text})'

    @staticmethod
    def _remove_parentheses(text):
        assert text[0] == '('
        assert text[1] == ')'
        return text[1:-1]

    def nl_token_to_action_name(self, nl_token):
        return self._add_parentheses(self._add_prefix(nl_token, self.nl_token_meta_name))

    def action_name_to_nl_token(self, action_name):
        return self._remove_parentheses(self._remove_prefix(action_name, self.nl_token_meta_name))

    def is_nl_token_action_name(self, action_name):
        return action_name[0] == '(' and action_name[-1] == ')' \
            and self._remove_parentheses(action_name).startswith(self.nl_token_meta_name + self._delimiter_after_prefix)

    def token_to_action_name(self, token, special_tokens):
        if token in special_tokens:
            return self.special_token_to_action_name(token)
        else:
            return self.nl_token_to_action_name(token)

    def action_name_to_token(self, action_name):
        if self.is_nl_token_action_name(action_name):
            return self.action_name_to_nl_token(action_name)
        else:
            return self.action_name_to_special_token(action_name)


class StrictTypeProcessing:
    @staticmethod
    def iter_super_to_sub_actions(super_types_dict, is_non_conceptual_type):
        from splogic.base.formalism import Action

        super_sub_pair_set = set()

        def find_super_to_sub_actions(sub_type, super_types):
            for super_type in super_types:
                if is_non_conceptual_type(super_type):
                    super_sub_pair_set.add((super_type, sub_type))
                    find_super_to_sub_actions(super_type, super_types_dict.get(super_type, []))
                else:
                    find_super_to_sub_actions(sub_type, super_types_dict.get(super_type, []))

        for sub_type, super_types in super_types_dict.items():
            if is_non_conceptual_type(sub_type):
                find_super_to_sub_actions(sub_type, super_types)

        super_sub_pairs = sorted(super_sub_pair_set)
        for super_type, sub_type in super_sub_pairs:
            yield Action(name=f'{super_type}-to-{sub_type}',
                         act_type=super_type,
                         param_types=[sub_type],
                         expr_dict=dict(default='{0}'))

    @staticmethod
    def get_strictly_typed_action_tree(grammar, action_name_tree, dynamic_binding={}):
        return StrictTypeProcessing.get_strictly_typed_action_seq(
            grammar=grammar, action_name_seq=flatten(action_name_tree),
            dynamic_binding=dynamic_binding, return_tree=True)

    @staticmethod
    def get_strictly_typed_action_seq(grammar, action_name_seq, dynamic_binding={}, return_tree=False):
        assert grammar.inferencing_subtypes is False

        input_action_seq = tuple(map(grammar.name_to_action, action_name_seq))
        num_processed_actions = 0
        output_action_seq = []

        def find_super_to_sub_type_seq(super_type, sub_type):
            type_seq_q = deque([typ] for typ in grammar.super_types_dict[sub_type])
            while type_seq_q:
                type_seq = type_seq_q.popleft()
                last_type = type_seq[-1]
                if super_type == last_type:
                    _type_seq = list(reversed([sub_type] + type_seq))
                    assert grammar.is_non_conceptual_type(_type_seq[0])
                    assert grammar.is_non_conceptual_type(_type_seq[-1])
                    type_seq = tuple(chain(
                        [_type_seq[0]],
                        filter(grammar.is_non_conceptual_type, _type_seq[1: -1]),
                        [_type_seq[-1]]))
                    return type_seq
                else:
                    type_seq_q.extend(type_seq + [typ] for typ in grammar.super_types_dict[last_type])
            else:
                raise Exception('Cannot find the type sequence')

        state = grammar.search_state_cls.create()
        while not state.tree.is_closed_root():
            expected_action = input_action_seq[num_processed_actions]
            num_processed_actions += 1
            # utterance_token_id_seq = grammar.utterance_tokenizer(question)['input_ids']
            # dynamic_trie = learning._utterance_token_id_seq_to_dynamic_trie(grammar, utterance_token_id_seq)
            # with grammar.let_dynamic_trie(dynamic_trie, using_spans_as_entities=True):
            #     candidate_action_ids = state.get_candidate_action_ids()
            with grammar.dynamic_scope.let(**dynamic_binding):
                candidate_action_ids = state.get_candidate_action_ids()
            if expected_action.id in candidate_action_ids:
                output_action_seq.append(expected_action)
                state = state.get_next_state(expected_action)
            else:
                opened_tree, children = state.tree.get_opened_tree_children()
                opened_action = opened_tree.value
                param_type = opened_action.param_types[grammar.formalism._get_next_param_idx(opened_action, len(children))]
                type_seq = find_super_to_sub_type_seq(param_type, expected_action.act_type)

                for idx in range(len(type_seq) - 1):
                    lhs_type = type_seq[idx]
                    rhs_type = type_seq[idx + 1]
                    action_name = f'{lhs_type}-to-{rhs_type}'
                    intermediate_action = grammar.name_to_action(action_name)
                    state = state.get_next_state(intermediate_action)
                    output_action_seq.append(intermediate_action)
                output_action_seq.append(expected_action)
                state = state.get_next_state(expected_action)

        if return_tree:
            assert len(state.tree.children) == 1
            # Assume state.tree.value is the start action, such as `program`.
            # The start action takes only one argument.

            output_action_tree = state.tree.children[0].get_value_tree()
            return output_action_tree
        else:
            return output_action_seq
