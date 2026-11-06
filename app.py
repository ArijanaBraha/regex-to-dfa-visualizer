"""
app.py - the interactive visualizer.

    python app.py            # opens http://127.0.0.1:8000 in the browser

Uses only the Python standard library plus the project's own `automata` engine. The page
(visualizer/index.html) asks this server for every stage of the pipeline as JSON and draws the
automata in the browser with Graphviz compiled to JavaScript (viz.js, bundled for offline use).
"""
import json
import mimetypes
import statistics
import sys
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from automata import DFA, NFA, EPSILON, RegexSyntaxError, StateLimitExceeded, parse, thompson
from automata.regex import Concat, Epsilon, Star, Symbol, Union, to_string

ROOT = Path(__file__).parent / 'visualizer'
SUBSET_LIMIT = 5000      # stop runaway determinizations
DRAW_LIMIT = 80          # larger automata are reported, not drawn


# ------------------------------------------------------------------ naming helpers
def letter_name(i: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, ..."""
    s = ''
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def bfs_order(dfa: DFA) -> list:
    """States in breadth-first order from q0, symbols in alphabetical order."""
    order, seen, queue = [], {dfa.q0}, deque([dfa.q0])
    while queue:
        s = queue.popleft()
        order.append(s)
        for c in sorted(dfa.S):
            t = dfa.d.get((s, c))
            if t is not None and t not in seen:
                seen.add(t)
                queue.append(t)
    return order


def set_label(members, name=str) -> str:
    if not members:
        return '∅'
    try:
        members = sorted(members)
    except TypeError:
        members = sorted(members, key=str)
    return '{' + ','.join(name(m) for m in members) + '}'


# ------------------------------------------------------------------ automata -> JSON
def automaton_json(aut, name, label=None, start_name=None):
    """Generic JSON for an NFA or DFA. `name(state)` gives the node id shown in the drawing."""
    edges = {}
    for (s, c), target in aut.d.items():
        for t in (target if isinstance(aut, NFA) else [target]):
            edges.setdefault((name(s), name(t)), []).append('ε' if c == EPSILON else c)
    states = [{'id': name(s), 'label': (label(s) if label else ''), 'final': s in aut.F}
              for s in sorted(aut.K, key=lambda s: str(name(s)))]
    return {
        'states': states,
        'start': start_name if start_name is not None else name(aut.q0),
        'edges': [{'from': a, 'to': b, 'label': ','.join(sorted(set(ls)))} for (a, b), ls in edges.items()],
        'n_states': len(aut.K),
        'n_transitions': sum(len(t) if isinstance(aut, NFA) else 1 for t in aut.d.values()),
        'drawable': len(aut.K) <= DRAW_LIMIT,
    }


def tree_json(node):
    match node:
        case Symbol(c):
            return {'label': c, 'kind': 'symbol', 'children': []}
        case Epsilon():
            return {'label': 'ε', 'kind': 'symbol', 'children': []}
        case Concat(l, r):
            return {'label': '·', 'kind': 'op', 'children': [tree_json(l), tree_json(r)]}
        case Union(l, r):
            return {'label': '|', 'kind': 'op', 'children': [tree_json(l), tree_json(r)]}
        case Star(x):
            return {'label': '*', 'kind': 'op', 'children': [tree_json(x)]}


def quotient_json(num: DFA, blocks, names):
    """Minimal DFA built from a final partition; each state is labelled by the merged states."""
    block_of = {s: i for i, block in enumerate(blocks) for s in block}
    q = DFA(S=num.S, K=set(range(len(blocks))), q0=block_of[num.q0],
            d={(block_of[s], c): block_of[t] for (s, c), t in num.d.items()},
            F={block_of[s] for s in num.F})
    order = {b: i for i, b in enumerate(bfs_order(q))}
    return automaton_json(q, lambda b: f'm{order[b]}',
                          lambda b: '{' + ','.join(names[s] for s in blocks[b]) + '}')


def named_blocks(trace, names):
    return [[[names[s] for s in block] for block in step['blocks']] for step in trace]


# ------------------------------------------------------------------ the pipeline
def median_ms(fn, repeats=5):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples)


def run(regex: str, alphabet: str | None):
    tree = parse(regex)
    nfa = thompson(tree, set(alphabet) if alphabet else None)

    subset_trace = []
    dfa = nfa.subset_construction(SUBSET_LIMIT, trace=subset_trace)
    order = bfs_order(dfa)
    idx = {s: i for i, s in enumerate(order)}
    names = [letter_name(i) for i in range(len(order))]
    num = dfa.remap_states(lambda s: idx[s])          # integer-labelled copy, same numbering

    result = {
        'regex': regex,
        'parsed': to_string(tree),
        'alphabet': sorted(nfa.S),
        'tree': tree_json(tree),
        'nfa': automaton_json(nfa, lambda s: str(s)),
        'dfa': automaton_json(num, lambda s: names[s], lambda s: set_label(order[s])),
        'subset_steps': [{'from': names[idx[a]], 'symbol': c, 'to': names[idx[b]],
                          'subset': set_label(b), 'new': new} for a, c, b, new in subset_trace],
        'algorithms': {},
    }

    # Moore and Hopcroft: partition snapshots + the resulting minimal DFA
    for algo in ('moore', 'hopcroft'):
        trace, ops = [], {}
        getattr(num, f'_{algo}_minimize')(ops, trace)
        final_blocks = trace[-1]['blocks']
        steps = []
        for i, step in enumerate(trace):
            entry = {'blocks': [[names[s] for s in b] for b in step['blocks']]}
            if 'splitter' in step:
                entry['splitter'] = [names[s] for s in step['splitter']]
                entry['symbol'] = step['symbol']
                entry['split'] = [[names[s] for s in part] for part in step['split']]
            steps.append(entry)
        result['algorithms'][algo] = {'steps': steps, 'ops': ops,
                                      'minimal': quotient_json(num, final_blocks, names)}

    # Brzozowski: the four intermediate automata
    try:
        r1 = num.reverse()
        d1 = r1.subset_construction(SUBSET_LIMIT)
        d1_order = bfs_order(d1)
        d1_idx = {s: i for i, s in enumerate(d1_order)}
        d1n = d1.remap_states(lambda s: d1_idx[s])
        r2 = d1n.reverse()
        d2 = r2.subset_construction(SUBSET_LIMIT)
        d2_order = bfs_order(d2)
        d2_idx = {s: i for i, s in enumerate(d2_order)}
        rev_name = lambda base: (lambda s: "S'" if s == '__REV_START__' else base(s))
        stages = [
            {'title': 'Kthimi i DFA-së', 'formula': 'rev(D)',
             'automaton': automaton_json(r1, rev_name(lambda s: names[s]))},
            {'title': 'Determinizimi i parë', 'formula': 'det(rev(D))',
             'automaton': automaton_json(d1n, lambda s: str(s + 1),
                                         lambda s: set_label([m for m in d1_order[s] if m != '__REV_START__'],
                                                             lambda m: names[m]))},
            {'title': 'Kthimi i dytë', 'formula': 'rev(det(rev(D)))',
             'automaton': automaton_json(r2, rev_name(lambda s: str(s + 1)))},
            {'title': 'Determinizimi i dytë = automati minimal', 'formula': 'det(rev(det(rev(D))))',
             'automaton': automaton_json(d2, lambda s: f'm{d2_idx[s]}',
                                         lambda s: set_label([m for m in s if m != '__REV_START__'],
                                                             lambda m: str(m + 1)))},
        ]
        result['algorithms']['brzozowski'] = {
            'stages': stages,
            'ops': {'intermediate_sizes': [len(r1.K), len(d1.K), len(r2.K), len(d2.K)]}}
    except StateLimitExceeded:
        result['algorithms']['brzozowski'] = {'error': f'Automati i ndërmjetëm i kaloi {SUBSET_LIMIT} gjendje.'}

    # comparison: the same calls as in the experiment (median of 5 runs)
    comparison = []
    for algo, fn in [('Moore', lambda o=None: dfa.minimize('moore', o)),
                     ('Hopcroft', lambda o=None: dfa.minimize('hopcroft', o)),
                     ('Brzozowski (DFA)', lambda o=None: dfa.minimize('brzozowski', o, SUBSET_LIMIT)),
                     ('Brzozowski (NFA)', lambda o=None: nfa.brzozowski_minimize(o, SUBSET_LIMIT))]:
        try:
            ops = {}
            minimal = fn(ops)
            comparison.append({'algorithm': algo, 'states': len(minimal.canonical().K),
                               'time_ms': round(median_ms(fn), 3), 'ops': ops})
        except StateLimitExceeded:
            comparison.append({'algorithm': algo, 'error': 'kufiri i gjendjeve u kalua'})
    canon = {json.dumps(sorted(map(str, fn().canonical().d.items())))
             for fn in [lambda: dfa.minimize('moore'), lambda: dfa.minimize('hopcroft')]}
    result['comparison'] = comparison
    result['all_equal'] = len(canon) == 1 and len({c.get('states') for c in comparison if 'states' in c}) == 1
    return result


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):          # keep the console quiet
        pass

    def send(self, status, body: bytes, ctype):
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/api/run':
            q = parse_qs(url.query)
            regex = q.get('regex', [''])[0]
            alphabet = q.get('alphabet', [''])[0] or None
            try:
                data = run(regex, alphabet)
                self.send(200, json.dumps(data, ensure_ascii=False).encode(), 'application/json; charset=utf-8')
            except RegexSyntaxError as e:
                self.send(400, json.dumps({'error': f'Gabim sintaksor: {e}'}, ensure_ascii=False).encode(),
                          'application/json; charset=utf-8')
            except ValueError as e:
                self.send(400, json.dumps({'error': str(e)}, ensure_ascii=False).encode(),
                          'application/json; charset=utf-8')
            except StateLimitExceeded:
                self.send(400, json.dumps({'error': f'DFA-ja i kalon {SUBSET_LIMIT} gjendje; provoni një '
                                                    f'shprehje më të vogël.'}, ensure_ascii=False).encode(),
                          'application/json; charset=utf-8')
            return
        rel = 'index.html' if url.path in ('', '/') else url.path.lstrip('/')
        path = (ROOT / rel).resolve()
        if ROOT.resolve() not in path.parents or not path.is_file():
            self.send(404, b'not found', 'text/plain')
            return
        ctype = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
        if path.suffix == '.woff2':
            ctype = 'font/woff2'
        self.send(200, path.read_bytes(), ctype)


def main():
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    port = int(nums[0]) if nums else 8000
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    url = f'http://127.0.0.1:{port}'
    print(f'Vizualizuesi po punon në {url}  (Ctrl+C për ta ndalur)')
    if '--no-browser' not in sys.argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nU ndal.')


if __name__ == '__main__':
    main()
