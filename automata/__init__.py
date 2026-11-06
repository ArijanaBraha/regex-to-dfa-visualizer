"""The regex -> NFA -> DFA -> minimal DFA engine."""
from .DFA import DFA
from .NFA import NFA, EPSILON, StateLimitExceeded
from .regex import parse, thompson, regex_to_nfa, RegexSyntaxError

ALGORITHMS = ("moore", "hopcroft", "brzozowski")
