import re
from itertools import chain
from collections import deque
from typing import List, Dict
from abc import ABCMeta, abstractmethod
import copy
import inspect
from functools import lru_cache, cache
from enum import Enum

from dhnamlib.pylib.structure import TreeStructure
from dhnamlib.pylib.iteration import any_not_none, flatten, split_by_indices, chainelems, lastelem, not_none_valued_pairs
# from dhnamlib.pylib.klass import Interface
from dhnamlib.pylib.klass import subclass, implement, override
# from dhnamlib.pylib.klass import abstractfunction
from dhnamlib.pylib.decoration import deprecated, unnecessary, construct, keyed_cache
from dhnamlib.pylib.structure import bidict, DuplicateValueError

from dhnamlib.hissplib.compile import eval_lissp
from dhnamlib.hissplib.macro import prelude


prelude()


# token
class Action:
    def __init__(self,
                 *,
                 name,
                 act_type,
                 param_types,
                 expr_dict,
                 optional_idx=None,
                 rest_idx=None,
                 arg_candidate=None,
                 arg_filter=None,
                 starting=False):
        self.name = name

        assert self.is_valid_act_type(act_type), f'"{act_type}" is not a valid act-type.'
        assert isinstance(param_types, (list, tuple))

        self.act_type = act_type
        self.param_types = param_types
        self.expr_dict = expr_dict
        self.expr_pieces_dict = self.get_expr_pieces_dict(expr_dict)
        self.optional_idx = optional_idx
        self.rest_idx = rest_idx
        self.arg_candidate = arg_candidate
        self.arg_filter = arg_filter
        self.num_min_args = self.get_min_num_args()
        self.starting = starting
        self._id = None

    @property
    def id(self):
        assert self._id is not None, 'action id is not set'
        return self._id

    @id.setter
    def id(self, id):
        assert self._id is None, 'id is already set'
        self._id = id

    _raw_left_curly_bracket_symbol = '___L_CURLY___'
    _raw_right_curly_bracket_symbol = '___R_CURLY___'
    _place_holder_regex = re.compile(r'{(([0-9]+)|([_a-zA-Z][_a-zA-Z0-9]*))}')

    class PieceKey:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return repr(self.value)

    @classmethod
    def get_expr_pieces_dict(cls, expr_dict):
        def replace_brackets_with_symbols(text):
            return text.replace(
                '{{', cls._raw_left_curly_bracket_symbol).replace(
                '}}', cls._raw_right_curly_bracket_symbol)

        def replace_symbols_with_brackets(text):
            return text.replace(
                cls._raw_left_curly_bracket_symbol, '{{').replace(
                cls._raw_right_curly_bracket_symbol, '}}')

        def iter_span_piece_key_pairs(expr):
            for match_obj in cls._place_holder_regex.finditer(expr):
                yield match_obj.span(), match_obj.group(1)

        def iter_expr_pieces(expr):
            span_piece_key_pairs = tuple(iter_span_piece_key_pairs(replace_brackets_with_symbols(expr)))
            if len(span_piece_key_pairs) == 0:
                yield expr
            else:
                spans, piece_keys = zip(*iter_span_piece_key_pairs(replace_brackets_with_symbols(expr)))
                splits = split_by_indices(expr, chainelems(spans), including_end_index=False)
                for idx, split in enumerate(splits):
                    if idx % 2 == 1:
                        yield Action.PieceKey(int(piece_keys[idx // 2]))
                    else:
                        yield replace_symbols_with_brackets(split)

        def get_expr_pieces(expr):
            if callable(expr):
                return expr
            else:
                assert isinstance(expr, str)
                return tuple(iter_expr_pieces(expr))

        return dict([piece_key, get_expr_pieces(expr)]
                    for piece_key, expr in expr_dict.items())

    @staticmethod
    def is_valid_act_type(act_type):
        if isinstance(act_type, tuple):
            return Action.is_valid_union_type(act_type)
        else:
            return isinstance(act_type, str)

    @staticmethod
    def is_valid_union_type(union_type):
        return len(union_type) > 1 and all(isinstance(typ, str) for typ in union_type)

    @staticmethod
    def is_union_type(typ):
        if isinstance(typ, tuple):
            assert Action.is_valid_union_type(typ)
            return True
        else:
            return False

    def has_union_act_type(self):
        return self.is_union_type(self.act_type)

    def __repr__(self):
        return self.name

    def has_param(self):
        return bool(self.param_types)

    @property
    def num_params(self):
        return len(self.param_types)

    @property
    def terminal(self):
        return not self.param_types

    def get_min_num_args(self):
        additional_idx = any_not_none([self.optional_idx, self.rest_idx], default=None)
        if additional_idx is not None:
            return additional_idx + 1
        else:
            return len(self.param_types)


class MetaAction:
    def __init__(self,
                 *,
                 meta_name,
                 meta_params,
                 name_fn,
                 expr_dict_fn,
                 **action_kwargs):
        self.meta_name = meta_name
        self.meta_params = meta_params
        self._meta_param_to_index = dict(map(reversed, enumerate(meta_params)))
        meta_action = self

        class SpecificAction(Action):
            def __init__(self, *, meta_args=None, meta_kwargs=None, **kwargs):
                assert meta_args is not None or meta_kwargs is not None, 'At least one of meta_args or meta_kwargs should be not None'

                self.meta_action = meta_action
                self.meta_args = () if meta_args is None else meta_args
                self.meta_kwargs = {} if meta_kwargs is None else meta_kwargs

                del meta_args, meta_kwargs

                assert len(self.meta_args) + len(self.meta_kwargs) == len(meta_action.meta_params)

                new_kwargs = dict(action_kwargs)
                for k, v in kwargs.items():
                    assert new_kwargs.get(k) is None
                    new_kwargs[k] = v

                assert 'name' not in new_kwargs
                new_kwargs['name'] = name_fn(*self.meta_args, **self.meta_kwargs)

                assert 'expr_dict' not in new_kwargs
                new_kwargs['expr_dict'] = expr_dict_fn(*self.meta_args, **self.meta_kwargs)

                super().__init__(**new_kwargs)

            def get_meta_arg(self, key):
                try:
                    if isinstance(key, int):
                        try:
                            return self.meta_arg[key]
                        except IndexError:
                            return self.meta_kwargs[meta_action.meta_params[key]]
                    else:
                        assert isinstance(key, str)
                        try:
                            return self.meta_kwargs[key]
                        except KeyError:
                            return self.meta_args[meta_action._meta_param_to_index[key]]
                except (IndexError, KeyError):
                    raise Exception(f'no value exists for key {key}')

        self.action_cls = SpecificAction

    @deprecated
    @staticmethod
    def get_func_num_args(func):
        return len(inspect.signature(func).parameters)

    def __repr__(self):
        return self.meta_name

    def __call__(self, *, meta_args=None, meta_kwargs=None, **kwargs):
        return self.action_cls(meta_args=meta_args, meta_kwargs=meta_kwargs, **kwargs)


class Formalism:
    """Formalism"""

    def __init__(self, default_expr_key='default', decoding_speed_optimization=True):
        self.default_expr_key = default_expr_key
        self.reduce_action = Action(name='reduce',
                                    act_type='reduce-type',
                                    param_types=[],
                                    expr_dict={self.default_expr_key: ''})
        self.decoding_speed_optimization = decoding_speed_optimization
        if not decoding_speed_optimization:
            self.get_allowed_and_ids_pairs = self._get_allowed_and_ids_pairs__non_optimized

    @staticmethod
    def check_action_name_overlap(actions, meta=False):
        """Check if a same name exists"""

        attr = 'meta_name' if meta else 'name'
        action_names = set()
        for action in actions:
            assert getattr(action, attr) not in action_names
            action_names.add(getattr(action, attr))
        del action_names

    @staticmethod
    def make_name_to_action_dict(actions, constructor=dict, meta=False):
        dic = {}
        Formalism.update_name_to_action_dict(dic, actions, meta=meta)
        if isinstance(dic, constructor):
            return dic
        else:
            return constructor(dic.items())

    @staticmethod
    def update_name_to_action_dict(name_to_action_dict, actions, meta=False):
        attr = 'meta_name' if meta else 'name'
        for action in actions:
            assert getattr(action, attr) not in name_to_action_dict
            name_to_action_dict[getattr(action, attr)] = action

    @staticmethod
    def name_to_action(name, name_to_action_dicts):
        action = any_not_none(name_to_action_dict.get(name)
                              for name_to_action_dict in name_to_action_dicts)
        return action

    @staticmethod
    def make_type_to_actions_dict(actions, super_types_dict, constructor=dict, to_id=False, inferencing_subtypes=True):
        dic = {}
        Formalism.update_type_to_actions_dict(dic, actions, super_types_dict, to_id=to_id, inferencing_subtypes=inferencing_subtypes)
        if isinstance(dic, constructor):
            return dic
        else:
            return constructor(dic.items())

    @staticmethod
    def update_type_to_actions_dict(type_to_actions_dict, actions, super_types_dict, to_id=False, inferencing_subtypes=True):
        for action in actions:
            type_q = deque(action.act_type if action.has_union_act_type() else
                           [action.act_type])
            while type_q:
                typ = type_q.popleft()
                type_to_actions_dict.setdefault(typ, set()).add(action.id if to_id else action)
                if inferencing_subtypes and typ in super_types_dict:
                    type_q.extend(super_types_dict[typ])

    @staticmethod
    def make_type_to_action_ids_dict(actions, super_types_dict, constructor=dict, inferencing_subtypes=True):
        return Formalism.make_type_to_actions_dict(
            actions, super_types_dict, constructor, to_id=True, inferencing_subtypes=inferencing_subtypes)

    @staticmethod
    def update_type_to_action_ids_dict(type_to_action_ids_dict, actions, super_types_dict, inferencing_subtypes=True):
        return Formalism.update_type_to_actions_dict(
            type_to_action_ids_dict, actions, super_types_dict, to_id=True, inferencing_subtypes=inferencing_subtypes)

    @staticmethod
    def make_id_to_action_dict(actions, constructor=dict):
        id_to_action_dict = constructor()
        Formalism.update_id_to_action_dict(id_to_action_dict, actions)
        return id_to_action_dict

    @staticmethod
    def update_id_to_action_dict(id_to_action_dict, actions):
        id_to_action_dict.update([action.id, action] for action in actions)

    @staticmethod
    def id_to_action(action_id, id_to_action_dicts):
        '''
        :raises NotFoundError: when no action corresponds to the input id
        '''
        action = any_not_none(id_to_action_dict.get(action_id)
                              for id_to_action_dict in id_to_action_dicts)
        return action

        # from dhnamlib.pylib.exception import NotFoundError
        # try:
        #     action = any_not_none(id_to_action_dict.get(action_id)
        #                           for id_to_action_dict in id_to_action_dicts)
        #     return action
        # except NotFoundError as e:
        #     breakpoint()
        #     print(e)

    @staticmethod
    def sub_and_super(super_types_dict, sub_type, super_type):
        type_q = deque(sub_type if Action.is_union_type(sub_type) else
                       [sub_type])
        assert not Action.is_union_type(super_type)

        while type_q:
            typ = type_q.popleft()
            if typ == super_type:
                return True
            else:
                if typ in super_types_dict:
                    type_q.extend(super_types_dict[typ])
        else:
            return False

    def _must_be_reduced(self, opened_tree, children):
        if len(children) > 0 and children[-1].value == self.reduce_action:
            return True
        elif opened_tree.value.rest_idx is not None:
            return False
        else:
            return len(children) == len(opened_tree.value.param_types)

    @staticmethod
    def _optionally_reducible(action, num_params):
        # it's called before an action is selected
        if action.rest_idx is not None:
            return num_params >= action.rest_idx
        else:
            assert num_params != action.num_params
            if action.optional_idx is not None:
                return num_params >= action.optional_idx
            else:
                return False

    @staticmethod
    def _get_next_param_idx(opened_action, current_num_args):
        rest_idx = opened_action.rest_idx
        if (rest_idx is not None) and (rest_idx < current_num_args):
            next_param_idx = rest_idx
        else:
            next_param_idx = current_num_args
        return next_param_idx

    @keyed_cache(lambda self, param_type, type_to_candidates_dicts, optionally_reducible, to_id: (
        param_type, tuple(sorted(map(id, type_to_candidates_dicts))), optionally_reducible, to_id))
    def _get_candidates_from_cache(self, param_type, type_to_candidates_dicts, optionally_reducible, to_id):
        additional_candidates = ([self.reduce_action.id if to_id else self.reduce_action] if optionally_reducible else [])
        return tuple(chain(
            *(type_to_actions_dict.get(param_type, []) for type_to_actions_dict in type_to_candidates_dicts),
            additional_candidates))

    def _get_candidates(self, opened_action, current_num_args, type_to_candidates_dicts, to_id=False):
        next_param_idx = self._get_next_param_idx(opened_action, current_num_args)
        param_type = opened_action.param_types[next_param_idx]

        optionally_reducible = Formalism._optionally_reducible(opened_action, current_num_args)
        candidates = self._get_candidates_from_cache(param_type, type_to_candidates_dicts, optionally_reducible, to_id)

        return candidates

    def get_candidate_action_ids(self, opened_action, current_num_args, type_to_action_ids_dicts):
        return self._get_candidates(opened_action, current_num_args, type_to_action_ids_dicts, to_id=True)

    @keyed_cache(lambda self, param_type, type_to_candidates_dicts, optionally_reducible, all_token_id_set: (
        param_type, tuple(sorted(map(id, type_to_candidates_dicts))), optionally_reducible, id(all_token_id_set)))
    def get_disallowed_ids(self, param_type, type_to_candidates_dicts, optionally_reducible, all_token_id_set):
        allowed_ids = self._get_candidates_from_cache(param_type, type_to_candidates_dicts, optionally_reducible, to_id=True)
        disallowed_ids = tuple(all_token_id_set.difference(allowed_ids))
        return disallowed_ids

    def get_allowed_and_ids_pairs(self, opened_action, current_num_args, type_to_candidates_dicts, all_token_id_set, threshold):
        next_param_idx = self._get_next_param_idx(opened_action, current_num_args)
        param_type = opened_action.param_types[next_param_idx]

        optionally_reducible = Formalism._optionally_reducible(opened_action, current_num_args)
        candidate_ids = self._get_candidates_from_cache(
            param_type, type_to_candidates_dicts, optionally_reducible, to_id=True)

        if len(candidate_ids) < threshold:
            return True, candidate_ids
        else:
            disallowed_ids = self.get_disallowed_ids(
                param_type, type_to_candidates_dicts, optionally_reducible, all_token_id_set)
            return False, disallowed_ids

    def _get_allowed_and_ids_pairs__non_optimized(self, opened_action, current_num_args, type_to_candidates_dicts, all_token_id_set, threshold):
        next_param_idx = self._get_next_param_idx(opened_action, current_num_args)
        param_type = opened_action.param_types[next_param_idx]

        optionally_reducible = Formalism._optionally_reducible(opened_action, current_num_args)
        candidate_ids = self._get_candidates_from_cache(
            param_type, type_to_candidates_dicts, optionally_reducible, to_id=True)

        return True, candidate_ids

    def extend_actions(self, actions, use_reduce=True):
        if use_reduce:
            default_actions = tuple(chain([self.reduce_action], actions))
        else:
            default_actions = tuple(actions)
        return default_actions

    @deprecated
    @staticmethod
    def make_name_to_id_dict(actions, start_id, constructor=dict, sorting=False):
        if sorting:
            names = sorted(set(action.name or action.name for action in actions))
        else:
            name_set = set()
            names = []
            for action in actions:
                if action.name not in name_set:
                    names.append(action.name)
                    name_set.add(action.name)
            assert len(name_set) == len(names)
        return constructor(map(reversed, enumerate(names, start_id)))

    @staticmethod
    def name_to_id(name, name_to_id_dicts):
        action_id = any_not_none(name_to_id_dict.get(name)
                                 for name_to_id_dict in name_to_id_dicts)
        return action_id

    @staticmethod
    def action_to_id_by_name(action, name_to_id_dicts):
        return any_not_none(name_to_id_dict.get(action.name)
                            for name_to_id_dict in name_to_id_dicts)

    @deprecated
    @staticmethod
    def set_action_id_by_name_to_id_dicts(action, name_to_id_dicts):
        action._id = Formalism.action_to_id_by_name(action, name_to_id_dicts)


# program tree
class ProgramTree(TreeStructure, metaclass=ABCMeta):
    @classmethod
    def create_root(cls, value, terminal=False):
        # TODO: check this method. type of output object
        tree = super(ProgramTree, cls).create_root(value, terminal)
        tree.min_num_actions = 1 + value.num_min_args
        tree.cur_num_actions = 1
        return tree

    @staticmethod
    @abstractmethod
    def get_formalism():
        pass

    @property
    def formalism(self):
        return self.get_formalism()

    def push_action(self, action):
        opened, children = self.get_opened_tree_children()
        if opened.value.num_min_args <= len(children):
            addend = 1 + action.num_min_args
        else:
            addend = action.num_min_args

        new_tree = self.push_term(action) if action.terminal else self.push_nonterm(action)
        new_tree.min_num_actions = self.min_num_actions + addend

        new_tree.cur_num_actions = self.cur_num_actions + 1

        return new_tree

    def reduce_with_children(self, children, value=None):
        new_tree = super().reduce_with_children(children, value)
        new_tree.min_num_actions = children[-1].min_num_actions if children else self.min_num_actions
        new_tree.cur_num_actions = children[-1].cur_num_actions if children else self.cur_num_actions
        return new_tree

    def reduce_tree_amap(self):
        "reduce tree as much as possible"
        tree = self
        while not tree.is_closed_root():
            opened_tree, children = tree.get_opened_tree_children()
            if self.formalism._must_be_reduced(opened_tree, children):
                tree = opened_tree.reduce_with_children(children)
            else:
                break
        return tree

    def subtree_size_ge(self, size):
        stack = [self]
        accum = 1
        if accum >= size:
            return True

        while stack:
            tree = stack.pop()
            for child in reversed(tree.children):
                accum += 1
                if accum >= size:
                    return True
                if not tree.terminal:
                    assert not tree.opened
                    stack.append(child)

        return False

    def get_reduced_subtrees(self):
        tree = self
        subtrees = []
        if tree.is_closed():
            while not tree.terminal:
                assert not tree.opened
                subtrees.append(tree)
                tree = tree.children[-1]
        return subtrees

    def get_expr_str(self, expr_key=None):
        if expr_key is None:
            expr_key = self.formalism.default_expr_key

        def get_expr_pieces(action):
            return any_not_none(
                action.expr_pieces_dict.get(k)
                for k in [expr_key, self.formalism.default_expr_key])

        def get_expr_form(tree):
            child_expr_forms = tuple([] if tree.terminal else map(get_expr_form, tree.children))
            expr_pieces_or_expr_fn = get_expr_pieces(tree.value)
            if callable(expr_pieces_or_expr_fn):
                expr_fn = expr_pieces_or_expr_fn
                expr_form = expr_fn(*map(form_to_str, child_expr_forms))
            else:
                expr_pieces = expr_pieces_or_expr_fn
                expr_form = []
                for expr_piece in expr_pieces:
                    if isinstance(expr_piece, Action.PieceKey):
                        expr_form.append(child_expr_forms[expr_piece.value])
                    else:
                        expr_form.append(expr_piece)
            return expr_form

        def form_to_str(expr_form):
            if isinstance(expr_form, str):
                return expr_form
            else:
                return ''.join(flatten(expr_form))

        return form_to_str(get_expr_form(self))


def make_program_tree_cls(formalism: Formalism, name=None, opening_cache_size=10000):
    @subclass
    class NewProgramTree(ProgramTree):
        # interface = Interface(ProgramTree)

        @override
        @lru_cache(maxsize=opening_cache_size)
        def get_opened_tree_children(self):
            if self.is_opened():
                opened_tree, children = self, tuple()
            else:
                opened_tree, siblings = self.prev.get_opened_tree_children()
                children = siblings + (self,)
            return opened_tree, children

        @implement
        @staticmethod
        def get_formalism():
            return formalism

    if name is not None:
        NewProgramTree.__name__ = NewProgramTree.__qualname__ = name

    return NewProgramTree

# search state
class SearchState(metaclass=ABCMeta):
    @classmethod
    def create(cls):
        state = cls()
        state.initialize()
        return state

    @classmethod
    def get_formalism(cls):
        return cls.get_program_tree_cls().get_formalism()

    @property
    def formalism(self):
        return self.get_formalism()

    @staticmethod
    @abstractmethod
    def get_program_tree_cls():
        pass

    @property
    def program_tree_cls(self):
        return self.get_program_tree_cls()

    def initialize(self):
        self.tree = self.program_tree_cls.create_root(self.get_start_action())
        for k, v in self.get_initial_attrs().items():
            setattr(self, k, v)

    @property
    def using_arg_candidate(self):
        return self.get_using_arg_candidate()

    @property
    def using_arg_filter(self):
        return self.get_using_arg_filter()

    @classmethod
    @abstractmethod
    def get_using_arg_candidate(cls):
        pass

    @classmethod
    @abstractmethod
    def get_using_arg_filter(cls):
        pass

    @abstractmethod
    def get_initial_attrs(self):
        pass

    @abstractmethod
    def get_start_action(self):
        pass

    def get_updated_tree(self, action):
        tree = self.tree.push_action(action).reduce_tree_amap()
        return tree

    def get_updated_state(self, updated_tree):
        state = copy.copy(self)
        state.tree = updated_tree
        for k, v in self.get_updated_attrs(updated_tree).items():
            setattr(state, k, v)

        return state

    def get_next_state(self, action):
        updated_tree = self.get_updated_tree(action)
        updated_state = self.get_updated_state(updated_tree)
        return updated_state

    @classmethod
    def _map_action_seq(cls, action_seq, *, initial_state=None, including_initial=False, including_candidate_ids,
                        including_allowed_and_ids_pairs=False, verifying=False):
        state = cls.create() if initial_state is None else initial_state

        if including_initial:
            yield dict(candidate_action_ids=None, state=state, allowed_and_ids_pairs=None)

        for action in action_seq:
            if including_candidate_ids or verifying:
                _candidate_action_ids = state.get_candidate_action_ids()
                if verifying and (action.id not in _candidate_action_ids):
                    # breakpoint()
                    raise InvalidCandidateActionError(f'{action} is not a candidate action in the current action tree {state.tree}')
                candidate_action_ids = _candidate_action_ids if including_candidate_ids else None
            else:
                candidate_action_ids = None

            if including_allowed_and_ids_pairs:
                allowed_and_ids_pairs = state.get_allowed_and_ids_pairs()
            else:
                allowed_and_ids_pairs = None

            # update state
            state = state.get_next_state(action)

            yield dict(not_none_valued_pairs(
                candidate_action_ids=candidate_action_ids,
                state=state,
                allowed_and_ids_pairs=allowed_and_ids_pairs))

    @classmethod
    @construct(tuple)
    def action_seq_to_state_seq(cls, action_seq, initial_state=None, including_initial=False, verifying=False):
        for info in cls._map_action_seq(
                action_seq, initial_state=initial_state,
                including_initial=including_initial,
                including_candidate_ids=False,
                including_allowed_and_ids_pairs=False,
                verifying=verifying):
            yield info['state']

    @classmethod
    @construct(tuple)
    def action_seq_to_candidate_action_ids_seq(cls, action_seq, initial_state=None, verifying=False):
        for info in cls._map_action_seq(
                action_seq, initial_state=initial_state,
                including_initial=False,
                including_candidate_ids=True,
                including_allowed_and_ids_pairs=False,
                verifying=verifying):
            yield info['candidate_action_ids']

    @classmethod
    @construct(tuple)
    def action_seq_to_allowed_and_ids_pairs_seq(cls, action_seq, initial_state=None, verifying=False):
        for info in cls._map_action_seq(
                action_seq, initial_state=initial_state,
                including_initial=False,
                including_candidate_ids=False,
                including_allowed_and_ids_pairs=True,
                verifying=verifying):
            yield info['allowed_and_ids_pairs']

    @classmethod
    def get_last_state(cls, action_seq, initial_state=None, verifying=False):
        return lastelem(cls.action_seq_to_state_seq(action_seq, initial_state=initial_state, verifying=verifying))

    @abstractmethod
    def get_updated_attrs(self, tree):
        pass

    @unnecessary
    def _get_candidate_actions(self):
        raise Exception('remove this method')
        opened_tree, children = self.tree.get_opened_tree_children()
        opened_action = opened_tree.value
        if opened_action.arg_candidate is None:
            actions = self.formalism._get_candidates(
                opened_action, len(children), self.get_type_to_actions_dicts())
        else:
            actions = opened_action.arg_candidate(self.tree)
        if opened_action.arg_filter is not None:
            actions = tuple(opened_action.arg_filter(self.tree, actions))
        return actions

    def get_candidate_action_ids(self):
        opened_tree, children = self.tree.get_opened_tree_children()
        opened_action = opened_tree.value
        if self.using_arg_candidate and opened_action.arg_candidate is not None:
            action_ids = opened_action.arg_candidate(self.tree)
        else:
            action_ids = self.formalism.get_candidate_action_ids(
                opened_action, len(children), self.get_type_to_action_ids_dicts())
        if self.using_arg_filter and opened_action.arg_filter is not None:
            action_ids = tuple(opened_action.arg_filter(self.tree, action_ids))

        return action_ids

    def get_allowed_and_ids_pairs(self):
        opened_tree, children = self.tree.get_opened_tree_children()
        opened_action = opened_tree.value
        if self.using_arg_candidate and opened_action.arg_candidate is not None:
            action_ids = opened_action.arg_candidate(self.tree)
            allowed = True
        else:
            all_token_id_set = self.get_all_token_id_set()
            threshold = len(all_token_id_set) // 2
            allowed, action_ids = self.formalism.get_allowed_and_ids_pairs(
                opened_action, len(children), self.get_type_to_action_ids_dicts(),
                all_token_id_set, threshold)
        if self.using_arg_filter and opened_action.arg_filter is not None:
            assert allowed
            action_ids = tuple(opened_action.arg_filter(self.tree, action_ids))

        return allowed, action_ids

    @deprecated
    @abstractmethod
    def get_type_to_actions_dicts(self):
        pass

    @abstractmethod
    def get_type_to_action_ids_dicts(self):
        pass

    @abstractmethod
    def get_all_token_id_set(self):
        pass
 
    @deprecated
    def _actions_to_ids_by_names(self, actions):
        name_to_id_dicts = self.get_name_to_id_dicts()

        def _action_to_id(action):
            return Formalism.action_to_id_by_name(action, name_to_id_dicts)

        return tuple(map(_action_to_id, actions))

    def _ids_to_actions(self, action_ids):
        raise NotImplementedError

    @abstractmethod
    def get_name_to_id_dicts(self):
        pass

    @deprecated
    def get_candidate_action_to_id_bidict(self):
        actions = self._get_candidate_actions()
        ids = self._actions_to_ids_by_names(actions)

        assert len(actions) == len(ids)

        action_to_id_bidict = bidict(zip(actions, ids))
        assert len(action_to_id_bidict) == len(actions)
        assert len(action_to_id_bidict.inverse) == len(ids)

        return action_to_id_bidict


class InvalidCandidateActionError(Exception):
    pass


def make_search_state_cls(
        grammar, name=None, using_arg_candidate=True, using_arg_filter=False, ids_to_mask_fn=None,
        extra_special_states=[]
):
    @subclass
    class BasicSearchState(SearchState):
        # interface = Interface(SearchState)
        _mask_cache = dict()

        SpecialState = Enum('SpecialState', tuple(chain(['INVALID', 'END'], extra_special_states)))

        @staticmethod
        @implement
        def get_program_tree_cls():
            return grammar.program_tree_cls

        @classmethod
        @implement
        def get_using_arg_candidate(cls):
            return using_arg_candidate

        @classmethod
        @implement
        def get_using_arg_filter(cls):
            return using_arg_filter

        @implement
        def get_initial_attrs(self):
            return dict()

        @implement
        def get_start_action(self):
            return grammar.start_action

        @implement
        def get_updated_attrs(self, tree):
            return dict()

        @deprecated
        @implement
        def get_type_to_actions_dicts(self):
            return grammar.get_type_to_actions_dicts()

        @implement
        def get_type_to_action_ids_dicts(self):
            return grammar.get_type_to_action_ids_dicts()

        @implement
        @classmethod
        @cache
        def get_all_token_id_set(cls):
            all_token_id_set = set(grammar.iter_all_token_ids())
            return all_token_id_set

        @implement
        def get_name_to_id_dicts(self):
            return grammar.get_name_to_id_dicts()

        @deprecated
        def _compute_mask_key_for_typed_candidates(self, opened_action, current_num_args, type_to_action_ids_dicts):
            assert opened_action.id is not None
            return (opened_action.id, current_num_args, tuple(map(id, type_to_action_ids_dicts)))

        @deprecated
        def get_candidate_action_id_mask(self):
            opened_tree, children = self.tree.get_opened_tree_children()
            opened_action = opened_tree.value

            using_arg_candidate = self.using_arg_candidate and opened_action.arg_candidate is not None
            using_arg_filter = self.using_arg_filter and opened_action.arg_filter is not None

            if (not using_arg_candidate) and (not using_arg_filter):
                args_for_typed_candidates = (opened_action, len(children), self.get_type_to_action_ids_dicts())
                cache_key = self._compute_mask_key_for_typed_candidates(*args_for_typed_candidates)
                if cache_key not in self._mask_cache:
                    action_ids = self.formalism.get_candidate_action_ids(*args_for_typed_candidates)
                    self._mask_cache[cache_key] = ids_to_mask_fn(action_ids)
                candidate_mask = self._mask_cache[cache_key]
            else:
                action_ids = self.get_candidate_action_ids()
                candidate_mask = ids_to_mask_fn(action_ids)

            return candidate_mask

    for special_state in BasicSearchState.SpecialState:
        setattr(BasicSearchState, special_state.name, special_state)

    if name is not None:
        BasicSearchState.__name__ = BasicSearchState.__qualname__ = name

    return BasicSearchState
