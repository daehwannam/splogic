
from functools import lru_cache
from itertools import chain

from hissp.munger import munge, demunge

from dhnamlib.pylib.lisp import (remove_comments, replace_prefixed_parens, is_keyword, keyword_to_symbol)
from dhnamlib.pylib.iteration import merge_dicts, chainelems
from dhnamlib.pylib.function import starloop  # imported for eval_lissp
from dhnamlib.pylib.decoration import Register, deprecated
from dhnamlib.pylib.klass import abstractfunction
# from dhnamlib.pylib.decoration import cache

from dhnamlib.hissplib.macro import prelude, load_macro
from dhnamlib.hissplib.module import import_lissp
from dhnamlib.hissplib.compile import eval_lissp
from dhnamlib.hissplib.expression import remove_backquoted_symbol_prefixes  # imported for eval_lissp
from dhnamlib.hissplib.operation import import_operators
from dhnamlib.hissplib.decoration import parse_hy_args, hy_function

from .formalism import Formalism, Action, MetaAction


prelude()  # used for eval_lissp
import_operators()  # used for eval_lissp

hissplib_basic = import_lissp('dhnamlib.hissplib.basic')
load_macro(hissplib_basic, 'el-let', 'let')

grammar_read_form = '(progn {})'


class Grammar:
    """Formal grammar"""

    def __init__(self, *, formalism, super_types_dict, actions, start_action, meta_actions, register,
                 is_non_conceptual_type=None, use_reduce=True, inferencing_subtypes=True):
        self.formalism = formalism
        self.super_types_dict = super_types_dict
        self.start_action = start_action
        self.inferencing_subtypes = inferencing_subtypes
        self.base_actions = formalism.extend_actions(actions, use_reduce=use_reduce)
        self.meta_actions = meta_actions
        self.register = register
        self.is_non_conceptual_type = is_non_conceptual_type
        self.added_actions = []

        # base actions
        self._name_to_base_action_dict = formalism.make_name_to_action_dict(self.base_actions)
        self._meta_name_to_meta_action_dict = formalism.make_name_to_action_dict(meta_actions, meta=True)
        self._type_to_base_actions_dict = formalism.make_type_to_actions_dict(
            self.base_actions, super_types_dict, inferencing_subtypes=inferencing_subtypes)
        self.start_action.id = self._start_action_id
        self._set_action_ids(self.base_actions)
        self._id_to_base_action_dict = formalism.make_id_to_action_dict(self.base_actions)
        self._type_to_base_action_ids_dict = formalism.make_type_to_action_ids_dict(
            self.base_actions, super_types_dict, inferencing_subtypes=inferencing_subtypes)

        # added actions
        self._name_to_added_action_dict = dict()
        self._type_to_added_actions_dict = dict()
        self._type_to_added_action_ids_dict = dict()
        self._id_to_added_action_dict = dict()

    @property
    def _start_action_id(self):
        return -1

    def _set_action_ids(self, actions):
        name_to_id_dicts = self.get_name_to_id_dicts()
        for action in actions:
            action.id = self.formalism.action_to_id_by_name(action, name_to_id_dicts)

    @property
    def reduce_action(self):
        return self.formalism.reduce_action

    def name_to_action(self, name):
        return self.formalism.name_to_action(name, self.get_name_to_action_dicts())

    def get_name_to_action_dicts(self):
        return [self._name_to_base_action_dict, self._name_to_added_action_dict]

    def meta_name_to_meta_action(self, meta_name):
        return self.formalism.name_to_action(meta_name, [self._meta_name_to_meta_action_dict])

    @deprecated
    def get_type_to_actions_dicts(self):
        return [self._type_to_base_actions_dict, self._type_to_added_actions_dict]

    def get_type_to_action_ids_dicts(self):
        return [self._type_to_base_action_ids_dict, self._type_to_added_action_ids_dict]

    def name_to_id(self, name):
        return self.formalism.name_to_id(name, self.get_name_to_id_dicts())

    @abstractfunction
    def get_name_to_id_dicts(self):
        pass

    def get_id_to_action_dicts(self):
        return [self._id_to_base_action_dict, self._id_to_added_action_dict]

    def id_to_action(self, action_id):
        '''
        :raises NotFoundError: when no action corresponds to the input id
        '''
        return self.formalism.id_to_action(action_id, self.get_id_to_action_dicts())

    def add_actions(self, actions):
        actions = tuple(actions)
        self.formalism.update_name_to_action_dict(self._name_to_added_action_dict, actions)
        self.added_actions.extend(actions)
        assert len(self._name_to_added_action_dict) == len(self.added_actions)

        self._set_action_ids(self.added_actions)

        self.formalism.update_type_to_actions_dict(
            self._type_to_added_actions_dict, actions, self.super_types_dict, inferencing_subtypes=self.inferencing_subtypes)
        self.formalism.update_type_to_action_ids_dict(
            self._type_to_added_action_ids_dict, actions, self.super_types_dict, inferencing_subtypes=self.inferencing_subtypes)
        self.formalism.update_id_to_action_dict(self._id_to_added_action_dict, actions)

    def sub_and_super(self, sub_type, super_type):
        return self.formalism.sub_and_super(self.super_types_dict, sub_type, super_type)

    @property
    def program_tree_cls(self):
        return self.get_program_tree_cls()

    @abstractfunction
    def get_program_tree_cls(self):
        pass

    @property
    def search_state_cls(self):
        return self.get_search_state_cls()

    @abstractfunction
    def get_search_state_cls(self):
        pass

    @property
    def compiler_cls(self):
        return self.get_compiler_cls()

    @abstractfunction
    def get_compiler_cls(self):
        pass

    @abstractfunction
    def iter_all_token_ids(self):
        raise NotImplementedError


def read_grammar(file_path, *, formalism=None, grammar_cls=Grammar):
    if formalism is None:
        formalism = Formalism()

    def parse_params(raw_params):
        munged_optional = munge('&optional')
        munged_rest = munge('&rest')

        optional_idx = None
        rest_idx = None
        params = []

        idx = 0
        for raw_param in raw_params:
            # if raw_param.endswith(munged_optional):
            if raw_param == munged_optional:
                assert optional_idx is None
                assert rest_idx is None
                optional_idx = idx
            # elif raw_param.endswith(munged_rest):
            elif raw_param == munged_rest:
                assert rest_idx is None
                rest_idx = idx
            else:
                params.append(demunge(raw_param))
                idx += 1

        return tuple(params), optional_idx, rest_idx

    def parse_act_type(raw_act_type):
        if isinstance(raw_act_type, str):
            return demunge(raw_act_type)
        else:
            assert isinstance(raw_act_type, (list, tuple))
            assert len(raw_act_type) > 1
            return tuple(map(demunge, raw_act_type))

    def make_define_types(super_types_dicts, types):
        def define_types(type_hierarchy_tuple):
            super_types_dict = {}

            def update_super_types_dict(parent_type, children):
                for child in children:
                    if isinstance(child, tuple):
                        child_kw, *grandchildren = child
                        assert is_keyword(child_kw)
                        child_type = keyword_to_symbol(child_kw)
                        update_super_types_dict(child_type, grandchildren)
                    else:
                        assert isinstance(child, str)
                        child_type = demunge(child)
                    super_types_dict.setdefault(child_type, set()).add(parent_type)

            root_kw, *children = type_hierarchy_tuple
            root_type = root_kw[1:]
            update_super_types_dict(root_type, children)

            super_types_dicts.append(super_types_dict)

            for sub_type, super_types in super_types_dict.items():
                types.add(sub_type)
                types.update(super_types)

        return define_types

    def make_declare_conceptual_types(conceptual_types):
        def declare_conceptual_types(types):
            for typ in map(demunge, types):
                if typ in conceptual_types:
                    raise Exception(f'"{typ}" is declared more than once.')
                else:
                    conceptual_types.add(typ)

        return declare_conceptual_types

    def make_define_action(actions, start_actions, is_non_conceptual_type):
        def define_action(name, act_type, param_types, expr_dict, **kwargs):
            parsed_param_types, optional_idx, rest_idx = parse_params(param_types)
            act_type = parse_act_type(act_type)

            assert all(map(is_non_conceptual_type, parsed_param_types))
            assert is_non_conceptual_type(act_type)

            action = Action(
                name=demunge(name),
                act_type=act_type,
                expr_dict=expr_dict,
                param_types=parsed_param_types,
                optional_idx=optional_idx,
                rest_idx=rest_idx,
                **kwargs)

            if action.starting:
                start_actions.append(action)
                if len(start_actions) > 1:
                    raise Exception('more than one starting action is defined')
            else:
                actions.append(action)
        return define_action

    def make_define_meta_action(meta_actions, is_non_conceptual_type):
        def define_meta_action(meta_name, *, meta_params, act_type=None, param_types=None, **kwargs):
            parsed_act_type = parse_act_type(act_type) if act_type is not None else None
            if param_types is None:
                parsed_param_types, optional_idx, rest_idx = None, None, None
            else:
                parsed_param_types, optional_idx, rest_idx = parse_params(param_types)

            assert parsed_param_types is None or all(map(is_non_conceptual_type, parsed_param_types))
            assert parsed_act_type is None or is_non_conceptual_type(parsed_act_type)

            meta_action = MetaAction(
                meta_name=demunge(meta_name),
                meta_params=tuple(map(demunge, meta_params)),
                act_type=parsed_act_type,
                param_types=parsed_param_types,
                optional_idx=optional_idx,
                rest_idx=rest_idx,
                **kwargs)

            meta_actions.append(meta_action)
        return define_meta_action

    def make_is_non_conceptual_type(types, conceptual_types):
        def is_non_conceptual_non_union_type(typ):
            assert isinstance(typ, str)
            return (typ in types) and (typ not in conceptual_types)

        def is_non_conceptual_type(typ):
            if isinstance(typ, tuple):
                return all(map(is_non_conceptual_non_union_type, typ))
            else:
                return is_non_conceptual_non_union_type(typ)

        return is_non_conceptual_type

    def make_dict(*symbols):
        args, kwargs = parse_hy_args(symbols)
        return dict(*args, **kwargs)

    def make_retrieve(register):
        def retrieve(key):
            if isinstance(key, str):
                new_key = demunge(key)
            else:
                new_key = tuple(map(demunge, key))
            return register.retrieve(new_key)

        return retrieve

    def preprocess_prefixed_parens(text):
        def expr_to_str(prefix, expr_repr):
            return 

        string_expr_prefix = '$'
        backquote = '`'
        return replace_prefixed_parens(
            text,
            info_dicts=[dict(prefix=string_expr_prefix, paren_pair='()',
                             fn=lambda x: '#"{}"'.format(x.replace('"', r'\"'))),
                        dict(prefix=backquote, paren_pair='()',
                             fn=lambda x: "(remove_backquoted_symbol_prefixes {}{})".format(backquote, x))])

    def make_grammar(text, preprocessing_prefixed_parens=True):
        text = remove_comments(text)
        if preprocessing_prefixed_parens:
            text = preprocess_prefixed_parens(text)
        text = grammar_read_form.format(text)

        super_types_dicts = []
        types = set()
        conceptual_types = set()
        actions = []
        start_actions = []
        meta_actions = []
        register = Register(strategy='conditional')

        is_non_conceptual_type = make_is_non_conceptual_type(types, conceptual_types)

        bindings = [['mapkv', make_dict],
                    ['define-types', make_define_types(super_types_dicts, types)],
                    ['declare-conceptual-types', make_declare_conceptual_types(conceptual_types)],
                    ['define-action', hy_function(make_define_action(actions, start_actions, is_non_conceptual_type))],
                    ['define-meta-action', hy_function(make_define_meta_action(meta_actions, is_non_conceptual_type))],
                    ['retrieve', make_retrieve(register)]]

        eval_result = eval_lissp(text, extra_ns=get_extra_ns(bindings))
        assert eval_result is None

        grammar = grammar_cls(
            formalism=formalism,
            super_types_dict=merge_dicts(super_types_dicts,
                                         merge_fn=lambda values: set(chainelems(values))),
            actions=actions,
            start_action=start_actions[0],
            meta_actions=meta_actions,
            register=register,
            is_non_conceptual_type=is_non_conceptual_type,
        )

        return grammar

    with open(file_path) as f:
        text = f.read()

    return make_grammar(text)


def get_extra_ns(bindings):
    return dict([munge(k), v] for k, v in bindings)


if __name__ == '__main__':
    # grammar = read_grammar('./logic/example.grammar')
    formalism = Formalism()
    grammar = read_grammar('./language/kopl/grammar.lissp', formalism=formalism)
    breakpoint()
    ()
