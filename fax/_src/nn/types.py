from dataclasses import dataclass
from enum import IntEnum
from functools import wraps
from reprlib import recursive_repr
from typing import Callable, Generic, TypeVar

import chex
import jax.lax
from jax.tree_util import register_pytree_node

Array = chex.Array
CondFunc = Callable[[Array], bool]
TransFunc = Callable[[Array], Array]

T = TypeVar("T")


def register_pytree_connect_overlay(cls):
    _flatten = lambda obj: ((obj.lhs, obj.rhs), None)

    def unflatten(d, children):
        cls_instance = cls(*children)
        return cls_instance

    _unflatten = unflatten
    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls


def register_graph_leaf(cls):
    _flatten = lambda obj: ((obj.v,), None)

    def unflatten(d, children):
        cls_instance = cls(*children)
        return cls_instance

    _unflatten = unflatten
    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls


@dataclass
class Graph(Generic[T]):

    def __mul__(self, other):
        return Connect(self, other)

    def __add__(self, other):
        return Overlay(self, other)


@register_graph_leaf
@dataclass
class Empty(Graph[T]):
    index: int = -1


@register_graph_leaf
@dataclass
class Vertex(Graph[T]):
    v: T


@register_pytree_connect_overlay
@dataclass
class Connect(Graph[T]):
    lhs: Graph[T]
    rhs: Graph[T]


@register_pytree_connect_overlay
@dataclass
class Overlay(Graph[T]):
    lhs: Graph[T]
    rhs: Graph[T]


class Functional:
    """
    Partial fork, add support for Currying and function composition.
    Because python function can received arbitrary number of parameters,
    __call__ try to evaluate the wrapped function only when *args and **kwargs are empty.
    In such case, __call__ will test if the closure is a least equal in size to the pos only,
    pos or kwords and kwords only arguments and the kwargs part of the closure contain a definition
    for all the kwords only arguments.
    This does not prevent malformed closures to be evaluated.
    __call__ with not empty arguments return a new Functional object with a new closure.

    TODO: trace defaults values

    """

    __slots__ = "func", "args", "keywords", "__dict__", "__weakref__"

    def __new__(cls, func, /, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if hasattr(func, "func"):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func

        self = super(Functional, cls).__new__(cls)

        self.func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(self, /, *args, **keywords):
        f_keywords = {**self.keywords, **keywords}
        f_args = self.args + args
        if len(args) + len(keywords) == 0:
            not_kwords_only_len = self.func.__code__.co_argcount
            kwords_only_len = self.func.__code__.co_kwonlyargcount
            kwords_only_var_names = self.func.__code__.co_varnames[
                                    not_kwords_only_len:not_kwords_only_len + kwords_only_len]

            if not_kwords_only_len + kwords_only_len <= len(f_args) + len(f_keywords) \
                    and all(item in f_keywords.keys() for item in kwords_only_var_names):
                return self.func(*f_args, **f_keywords)
            else:
                return self
        else:
            return Functional(self.func, *f_args, **f_keywords)

    def __mul__(self, other):
        if hasattr(other, "func"):
            other_func = other.func
            other_args = other.args
            other_kwargs = other.keywords
        else:
            other_args = ()
            other_kwargs = {}
            other_func = other

        @wraps(other_func)
        def inner(*args, **kwargs):
            f_args = other_func(*args, **kwargs)
            f = self.func
            inner_args = self.args
            inner_kwargs = self.keywords
            if not isinstance(f_args, tuple):
                f_args = (f_args,)
            f_args = inner_args + f_args

            return f(*f_args, **inner_kwargs)

        return Functional(inner, *other_args, **other_kwargs)

    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.func)]
        args.extend(repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.keywords.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return type(self), (self.func,), (self.func, self.args,
                                          self.keywords or None, self.__dict__ or None)

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 5 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (not callable(func) or not isinstance(args, tuple) or
                (kwds is not None and not isinstance(kwds, dict)) or
                (namespace is not None and not isinstance(namespace, dict))):
            raise TypeError("invalid partial state")

        args = tuple(args)  # just in case it's a subclass
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict:  # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.func = func
        self.args = args
        self.keywords = kwds


register_pytree_node(
    Functional,
    lambda partial_: ((partial_.args, partial_.keywords), partial_.func),
    lambda func, xs: Functional(func, *xs[0], **xs[1]),
)


class RunMode(IntEnum):
    Train = 1
    Test = 2


class Normalization(IntEnum):
    Min_max = 1
    Mean_var = 2


class LocalRule(IntEnum):
    Oja = 1
    Reversed_Oja = 2
    Hebbian = 3
class WordEncoding(IntEnum):
    BOW = 1
    Learned = 2
