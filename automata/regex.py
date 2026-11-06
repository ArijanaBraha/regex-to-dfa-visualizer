"""
regex.py - stage 1 of the pipeline:  regular expression -> syntax tree -> epsilon-NFA

Supported syntax
    a b c ...     single characters (any character that is not special)
    xy            concatenation
    x|y           union
    x*            Kleene star          x+  one or more (= x x*)     x?  optional (= x|ε)
    ( )           grouping             ()  or  ε  the empty word
    \\c           escape: the character c literally (e.g. \\* or \\|)

Precedence, from strongest to weakest: * + ?   then concatenation   then |

Theory: Thompson (1968); Aho, Lam, Sethi & Ullman (2006) sec. 3.7.4.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from .NFA import NFA, EPSILON

sys.setrecursionlimit(10_000)


class RegexSyntaxError(ValueError):
    pass


# ------------------------------------------------------------------
# Syntax tree
# ------------------------------------------------------------------

@dataclass(frozen=True)
class Symbol:
    char: str


@dataclass(frozen=True)
class Epsilon:
    pass


@dataclass(frozen=True)
class Concat:
    left: 'Regex'
    right: 'Regex'


@dataclass(frozen=True)
class Union:
    left: 'Regex'
    right: 'Regex'


@dataclass(frozen=True)
class Star:
    inner: 'Regex'


Regex = Symbol | Epsilon | Concat | Union | Star


# ------------------------------------------------------------------
# Parser (recursive descent)
#
#   union  := concat ('|' concat)*
#   concat := repeat*                      (empty -> ε)
#   repeat := atom ('*' | '+' | '?')*
#   atom   := char | '\' char | 'ε' | '(' union ')'
# ------------------------------------------------------------------

class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def peek(self) -> str | None:
        return self.text[self.pos] if self.pos < len(self.text) else None

    def parse(self) -> Regex:
        tree = self.union()
        if self.pos != len(self.text):
            raise RegexSyntaxError(f"unexpected {self.text[self.pos]!r} at position {self.pos}")
        return tree

    def union(self) -> Regex:
        node = self.concat()
        while self.peek() == '|':
            self.pos += 1
            node = Union(node, self.concat())
        return node

    def concat(self) -> Regex:
        node = None
        while self.peek() is not None and self.peek() not in '|)':
            part = self.repeat()
            node = part if node is None else Concat(node, part)
        return node if node is not None else Epsilon()

    def repeat(self) -> Regex:
        node = self.atom()
        while self.peek() is not None and self.peek() in '*+?':
            op = self.text[self.pos]
            self.pos += 1
            if op == '*':
                node = Star(node)
            elif op == '+':
                node = Concat(node, Star(node))
            else:
                node = Union(node, Epsilon())
        return node

    def atom(self) -> Regex:
        c = self.peek()
        if c is None:
            raise RegexSyntaxError("unexpected end of regular expression")
        if c == '(':
            self.pos += 1
            node = self.union()
            if self.peek() != ')':
                raise RegexSyntaxError(f"missing ')' at position {self.pos}")
            self.pos += 1
            return node
        if c == ')':
            raise RegexSyntaxError(f"unmatched ')' at position {self.pos}")
        if c in '*+?':
            raise RegexSyntaxError(f"nothing to repeat at position {self.pos}")
        if c == '\\':
            if self.pos + 1 >= len(self.text):
                raise RegexSyntaxError("regular expression ends with a lone backslash")
            self.pos += 2
            return Symbol(self.text[self.pos - 1])
        self.pos += 1
        return Epsilon() if c == 'ε' else Symbol(c)


def parse(regex: str) -> Regex:
    """Parse a regular expression into a syntax tree."""
    return _Parser(regex).parse()


def alphabet_of(tree: Regex) -> set[str]:
    """All symbols that occur in the expression."""
    match tree:
        case Symbol(c):
            return {c}
        case Epsilon():
            return set()
        case Concat(l, r) | Union(l, r):
            return alphabet_of(l) | alphabet_of(r)
        case Star(x):
            return alphabet_of(x)


def to_string(tree: Regex) -> str:
    """Print a syntax tree back as a fully parenthesised regex (useful for debugging)."""
    match tree:
        case Symbol(c):
            return '\\' + c if c in '|*+?()\\ε' else c
        case Epsilon():
            return 'ε'
        case Concat(l, r):
            return f"({to_string(l)}{to_string(r)})"
        case Union(l, r):
            return f"({to_string(l)}|{to_string(r)})"
        case Star(x):
            return f"({to_string(x)})*"


# ------------------------------------------------------------------
# Thompson's construction
# ------------------------------------------------------------------

def thompson(tree: Regex, alphabet: set[str] | None = None) -> NFA[int]:
    """Thompson (1968). Every sub-expression becomes a fragment with exactly one start state
    and one accepting state; fragments are glued together with epsilon transitions.
    The result has at most 2 * (number of symbols and operators) states."""
    symbols = alphabet_of(tree)
    if alphabet is None:
        alphabet = symbols
    elif not symbols <= set(alphabet):
        raise ValueError(f"symbols {symbols - set(alphabet)} are not in the given alphabet")

    d: dict[tuple[int, str], set[int]] = {}
    count = 0

    def new_state() -> int:
        nonlocal count
        count += 1
        return count - 1

    def add(p: int, c: str, q: int) -> None:
        d.setdefault((p, c), set()).add(q)

    def build(node: Regex) -> tuple[int, int]:
        match node:
            case Symbol(c):                     # s --c--> f
                s, f = new_state(), new_state()
                add(s, c, f)
                return s, f
            case Epsilon():                     # s --ε--> f
                s, f = new_state(), new_state()
                add(s, EPSILON, f)
                return s, f
            case Concat(l, r):                  # [l] --ε--> [r]
                s1, f1 = build(l)
                s2, f2 = build(r)
                add(f1, EPSILON, s2)
                return s1, f2
            case Union(l, r):                   # s --ε--> [l] / [r] --ε--> f
                s, f = new_state(), new_state()
                for part in (l, r):
                    si, fi = build(part)
                    add(s, EPSILON, si)
                    add(fi, EPSILON, f)
                return s, f
            case Star(x):                       # s --ε--> [x] --ε--> f, loop back, skip
                s, f = new_state(), new_state()
                si, fi = build(x)
                add(s, EPSILON, si)
                add(s, EPSILON, f)
                add(fi, EPSILON, si)
                add(fi, EPSILON, f)
                return s, f

    start, accept = build(tree)
    return NFA(S=set(alphabet), K=set(range(count)), q0=start, d=d, F={accept})


def regex_to_nfa(regex: str, alphabet: set[str] | None = None) -> NFA[int]:
    """Stage 1 of the pipeline: regex string -> epsilon-NFA."""
    return thompson(parse(regex), alphabet)
