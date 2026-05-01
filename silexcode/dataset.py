from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Any, Callable


MASK64 = (1 << 64) - 1
GLOBAL_SEED = 0x53494C4558434F44
PUBLIC_EXAMPLES = 4

COND_SCALAR = [
    "a<b",
    "a<=t",
    "a+b<t",
    "a%max(abs(b),1)==0",
]

EXPR_SCALAR = [
    "a+b+c1",
    "a-b+c1",
    "b-a+c1",
    "a*t+b",
    "abs(a-b)+c1",
    "a*a+b",
    "min(a,b)+t",
    "max(a,b)-t",
]

EXPR_VEC = [
    "x[i]",
    "x[i]+c0",
    "x[i]-c0",
    "x[i]*c0",
    "abs(x[i])",
    "x[i]*x[i]",
    "x[i]+i",
    "x[i]-t",
]

EXPR_VEC_SEXPR = [
    "(at x i)",
    "(+ (at x i) c0)",
    "(- (at x i) c0)",
    "(* (at x i) c0)",
    "(abs (at x i))",
    "(* (at x i) (at x i))",
    "(+ (at x i) i)",
    "(- (at x i) t)",
]

PRED_VEC = [
    "1",
    "x[i]<t",
    "x[i]<=t",
    "x[i]==t",
    "x[i]!=t",
    "x[i]>t",
    "x[i]>=0",
    "x[i]%m==r",
    "i%m==r",
]

PRED_VEC_SEXPR = [
    "1",
    "(< (at x i) t)",
    "(<= (at x i) t)",
    "(= (at x i) t)",
    "(!= (at x i) t)",
    "(> (at x i) t)",
    "(>= (at x i) 0)",
    "(= (% (at x i) m) r)",
    "(= (% i m) r)",
]

PRED_PAIR = [
    "x[i]<x[j]",
    "x[i]+x[j]==t",
    "abs(x[i]-x[j])<=abs(t)",
    "(x[i]+x[j])%m==r",
]

C0_VALUES = [-3, -2, -1, 1, 2, 3]
C1_VALUES = [-7, -3, 0, 3, 7]
M_VALUES = [2, 3, 4, 5, 6, 7]

FAMILY_WEIGHTS = {
    1: [(300, 0), (250, 1), (200, 2), (150, 3), (100, 4)],
    2: [(100, 0), (150, 1), (200, 2), (150, 3), (150, 4), (150, 5), (100, 6)],
    3: [(80, 0), (110, 1), (160, 2), (120, 3), (120, 4), (130, 5), (120, 6), (100, 7), (60, 8)],
}

ALLOWED_AST_NODES = {
    "Module", "FunctionDef", "arguments", "arg", "Return", "Assign", "If", "For", "While",
    "Name", "Load", "Store", "Constant", "BinOp", "UnaryOp", "BoolOp", "Compare", "Call",
    "Subscript", "Slice", "List", "Mult", "Add", "Sub", "Mod", "BitXor", "BitAnd", "BitOr",
    "FloorDiv", "USub", "And", "Or", "Not", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE",
}
ALLOWED_CALLS = {"range", "len", "abs", "min", "max"}
FORBIDDEN_CALLS = {
    "eval", "exec", "open", "compile", "globals", "locals", "__import__", "getattr",
    "setattr", "delattr", "input", "print",
}
TRACE_VARS = set("abcijkmnqrtxy")


def splitmix64_next(x: int) -> tuple[int, int]:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = z ^ (z >> 31)
    return x, z & MASK64


class RNG:
    def __init__(self, seed: int):
        self.state = seed & MASK64

    def u64(self) -> int:
        self.state, z = splitmix64_next(self.state)
        return z

    def randint(self, lo: int, hi: int) -> int:
        return lo + (self.u64() % (hi - lo + 1))

    def choice(self, xs: list):
        return xs[self.u64() % len(xs)]

    def weighted_choice_1000(self, pairs: list[tuple[int, object]]):
        u = self.u64() % 1000
        acc = 0
        for w, x in pairs:
            acc += w
            if u < acc:
                return x
        raise RuntimeError("WEIGHTS_DO_NOT_SUM_TO_1000")


def record_seed(stage: int, index: int, nonce: int) -> int:
    return GLOBAL_SEED ^ (stage << 56) ^ (index << 8) ^ nonce


def assert_ascii(s: str) -> None:
    for ch in s:
        if ord(ch) > 127:
            raise ValueError("NON_ASCII_BYTE_FORBIDDEN")


def encode_ascii_record(s: str) -> list[int]:
    assert_ascii(s)
    return [256] + [ord(ch) for ch in s] + [257]


def encode_ascii_record_without_eos(s: str) -> list[int]:
    assert_ascii(s)
    return [256] + [ord(ch) for ch in s]


def serialize_value(v: Any) -> str:
    if isinstance(v, list):
        return "[" + ",".join(serialize_value(x) for x in v) + "]"
    if isinstance(v, tuple):
        return "(" + ",".join(serialize_value(x) for x in v) + ")"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    raise TypeError(f"unsupported value: {type(v).__name__}")


def render_F0(cid: int, sid_true: int, sid_false: int, c1: int = 0) -> str:
    return (
        "def f(a,b,t):\n"
        f"    c1={c1}\n"
        f"    if {COND_SCALAR[cid]}:\n"
        f"        return {EXPR_SCALAR[sid_true]}\n"
        f"    return {EXPR_SCALAR[sid_false]}\n"
    )


def render_F1(eid: int, c0: int = 1, m: int = 2, r: int = 0) -> str:
    return (
        "def f(x,n,t):\n"
        f"    c0={c0}\n"
        f"    m={m}\n"
        f"    r={r}\n"
        "    y=[0]*n\n"
        "    for i in range(n):\n"
        f"        y[i]={EXPR_VEC[eid]}\n"
        "    return y\n"
    )


def render_F2(pid: int, eid: int, c0: int = 1, m: int = 2, r: int = 0) -> str:
    return (
        "def f(x,n,t):\n"
        f"    c0={c0}\n"
        f"    m={m}\n"
        f"    r={r}\n"
        "    r0=0\n"
        "    for i in range(n):\n"
        f"        if {PRED_VEC[pid]}:\n"
        f"            r0=r0+{EXPR_VEC[eid]}\n"
        "    return r0\n"
    )


def render_F3(pid: int, m: int = 2, r: int = 0) -> str:
    return (
        "def f(x,n,t):\n"
        f"    m={m}\n"
        f"    r={r}\n"
        "    r0=0\n"
        "    for i in range(n):\n"
        f"        if {PRED_VEC[pid]}:\n"
        "            r0=r0+1\n"
        "    return r0\n"
    )


def render_F4(pid: int, m: int = 2, r: int = 0) -> str:
    return (
        "def f(x,n,t):\n"
        f"    m={m}\n"
        f"    r={r}\n"
        "    i=0\n"
        f"    while i<n and not ({PRED_VEC[pid]}):\n"
        "        i=i+1\n"
        "    return i\n"
    )


def render_F5(qid: int, m: int = 2, r: int = 0) -> str:
    return (
        "def f(x,n,t):\n"
        f"    m={m}\n"
        f"    r={r}\n"
        "    r0=0\n"
        "    for i in range(n):\n"
        "        for j in range(i+1,n):\n"
        f"            if {PRED_PAIR[qid]}:\n"
        "                r0=r0+1\n"
        "    return r0\n"
    )


def render_F6(pid: int, eid: int, c0: int = 1, m: int = 2, r: int = 0) -> str:
    return (
        "def f(x,n,t):\n"
        f"    c0={c0}\n"
        f"    m={m}\n"
        f"    r={r}\n"
        "    y=[0]*n\n"
        "    r0=0\n"
        "    for i in range(n):\n"
        f"        if {PRED_VEC[pid]}:\n"
        f"            r0=r0+{EXPR_VEC[eid]}\n"
        "        y[i]=r0\n"
        "    return y\n"
    )


def render_F7(direction: int) -> str:
    cmp_expr = "y[j]<y[k]" if direction == 0 else "y[j]>y[k]"
    return (
        "def f(x,n,t):\n"
        "    y=x[:n]\n"
        "    for i in range(n):\n"
        "        k=i\n"
        "        for j in range(i+1,n):\n"
        f"            if {cmp_expr}:\n"
        "                k=j\n"
        "        q=y[i]\n"
        "        y[i]=y[k]\n"
        "        y[k]=q\n"
        "    return y\n"
    )


def render_F8() -> str:
    return (
        "def f(n,t):\n"
        "    if n<=1:\n"
        "        return 1\n"
        "    a=1\n"
        "    b=1\n"
        "    i=2\n"
        "    while i<=n:\n"
        "        c=a+b\n"
        "        a=b\n"
        "        b=c\n"
        "        i=i+1\n"
        "    return b\n"
    )


def ast_whitelist_ok(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        name = type(node).__name__
        if name in {"Import", "ImportFrom", "Attribute"}:
            return False
        if name not in ALLOWED_AST_NODES:
            return False
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return False
            if node.func.id in FORBIDDEN_CALLS or node.func.id not in ALLOWED_CALLS:
                return False
    return True


def compile_restricted_function(code: str) -> Callable:
    if not ast_whitelist_ok(code):
        raise ValueError("INVALID_RESTRICTED_AST")
    env = {"__builtins__": {"range": range, "len": len, "abs": abs, "min": min, "max": max}}
    loc: dict[str, Any] = {}
    exec(compile(ast.parse(code), "<synthetic>", "exec"), env, loc)
    fn = loc.get("f")
    if not callable(fn):
        raise ValueError("NO_FUNCTION_F")
    return fn


def run_restricted(fn: Callable, case: tuple) -> Any:
    return fn(*case)


def gen_vec_case(rng: RNG, stage: int, case_index: int) -> tuple[list[int], int, int]:
    n_max = {1: 4, 2: 8, 3: 32}[stage]
    lo = {1: -9, 2: -31, 3: -127}[stage]
    hi = {1: 9, 2: 31, 3: 127}[stage]
    if case_index == 0:
        return ([], 0, 0)
    if case_index == 1:
        return ([lo], 1, hi)
    if case_index == 2:
        return ([0 for _ in range(n_max)], n_max, 0)
    if case_index == 3:
        return ([lo + ((hi - lo) * i) // max(1, n_max - 1) for i in range(n_max)], n_max, 0)
    n = rng.randint(0, n_max)
    return ([rng.randint(lo, hi) for _ in range(n)], n, rng.randint(lo, hi))


def gen_scalar_case(rng: RNG, stage: int, case_index: int) -> tuple[int, int, int]:
    lo = {1: -9, 2: -31, 3: -127}[stage]
    hi = {1: 9, 2: 31, 3: 127}[stage]
    if case_index == 0:
        return (0, 0, 0)
    if case_index == 1:
        return (lo, hi, 0)
    if case_index == 2:
        return (hi, lo, hi)
    if case_index == 3:
        return (1, -1, lo)
    return (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))


def gen_dp_case(rng: RNG, case_index: int) -> tuple[int, int]:
    if case_index == 0:
        return (0, 0)
    if case_index == 1:
        return (1, 0)
    if case_index == 2:
        return (2, 0)
    if case_index == 3:
        return (24, 0)
    return (rng.randint(0, 24), 0)


def make_cases(family_id: int, stage: int, rng: RNG, count: int) -> list[tuple]:
    cases = []
    for i in range(count):
        if family_id == 0:
            cases.append(gen_scalar_case(rng, stage, i))
        elif family_id == 8:
            cases.append(gen_dp_case(rng, i))
        else:
            cases.append(gen_vec_case(rng, stage, i))
    return cases


def sample_params(family_id: int, rng: RNG) -> dict[str, int]:
    if family_id == 0:
        return {"cid": rng.randint(0, 3), "sid_false": rng.randint(0, 7), "sid_true": rng.randint(0, 7), "c1": rng.choice(C1_VALUES)}
    if family_id == 1:
        return {"eid": rng.randint(0, 7), "c0": rng.choice(C0_VALUES), "m": rng.choice(M_VALUES), "r": 0}
    if family_id in (2, 6):
        m = rng.choice(M_VALUES)
        return {"eid": rng.randint(0, 7), "pid": rng.randint(0, 8), "c0": rng.choice(C0_VALUES), "m": m, "r": rng.randint(0, m - 1)}
    if family_id in (3, 4):
        m = rng.choice(M_VALUES)
        return {"pid": rng.randint(0, 8), "m": m, "r": rng.randint(0, m - 1)}
    if family_id == 5:
        m = rng.choice(M_VALUES)
        return {"qid": rng.randint(0, 3), "m": m, "r": rng.randint(0, m - 1)}
    if family_id == 7:
        return {"direction": rng.randint(0, 1)}
    if family_id == 8:
        return {}
    raise ValueError("INVALID_FAMILY")


def render_code(family_id: int, params: dict[str, int]) -> str:
    if family_id == 0:
        return render_F0(params["cid"], params["sid_true"], params["sid_false"], params["c1"])
    if family_id == 1:
        return render_F1(params["eid"], params["c0"], params["m"], params["r"])
    if family_id == 2:
        return render_F2(params["pid"], params["eid"], params["c0"], params["m"], params["r"])
    if family_id == 3:
        return render_F3(params["pid"], params["m"], params["r"])
    if family_id == 4:
        return render_F4(params["pid"], params["m"], params["r"])
    if family_id == 5:
        return render_F5(params["qid"], params["m"], params["r"])
    if family_id == 6:
        return render_F6(params["pid"], params["eid"], params["c0"], params["m"], params["r"])
    if family_id == 7:
        return render_F7(params["direction"])
    if family_id == 8:
        return render_F8()
    raise ValueError("INVALID_FAMILY")


def eval_expr_vec(eid: int, x: list[int], i: int, t: int, params: dict[str, int]) -> int:
    c0 = params.get("c0", 1)
    if eid == 0:
        return x[i]
    if eid == 1:
        return x[i] + c0
    if eid == 2:
        return x[i] - c0
    if eid == 3:
        return x[i] * c0
    if eid == 4:
        return abs(x[i])
    if eid == 5:
        return x[i] * x[i]
    if eid == 6:
        return x[i] + i
    if eid == 7:
        return x[i] - t
    raise ValueError("INVALID_EID")


def eval_pred_vec(pid: int, x: list[int], i: int, t: int, params: dict[str, int]) -> bool:
    m = params.get("m", 2)
    r = params.get("r", 0)
    if pid == 0:
        return True
    if pid == 1:
        return x[i] < t
    if pid == 2:
        return x[i] <= t
    if pid == 3:
        return x[i] == t
    if pid == 4:
        return x[i] != t
    if pid == 5:
        return x[i] > t
    if pid == 6:
        return x[i] >= 0
    if pid == 7:
        return x[i] % m == r
    if pid == 8:
        return i % m == r
    raise ValueError("INVALID_PID")


def run_reference_interpreter(family_id: int, params: dict[str, int], case: tuple) -> Any:
    if family_id == 0:
        a, b, t = case
        c1 = params["c1"]
        sid = params["sid_true"] if [a < b, a <= t, a + b < t, a % max(abs(b), 1) == 0][params["cid"]] else params["sid_false"]
        return [
            a + b + c1,
            a - b + c1,
            b - a + c1,
            a * t + b,
            abs(a - b) + c1,
            a * a + b,
            min(a, b) + t,
            max(a, b) - t,
        ][sid]
    if family_id == 8:
        n, _t = case
        if n <= 1:
            return 1
        a = b = 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    x, n, t = case
    if family_id == 1:
        return [eval_expr_vec(params["eid"], x, i, t, params) for i in range(n)]
    if family_id == 2:
        return sum(eval_expr_vec(params["eid"], x, i, t, params) for i in range(n) if eval_pred_vec(params["pid"], x, i, t, params))
    if family_id == 3:
        return sum(1 for i in range(n) if eval_pred_vec(params["pid"], x, i, t, params))
    if family_id == 4:
        for i in range(n):
            if eval_pred_vec(params["pid"], x, i, t, params):
                return i
        return n
    if family_id == 5:
        qid = params["qid"]
        m = params["m"]
        r = params["r"]
        out = 0
        for i in range(n):
            for j in range(i + 1, n):
                ok = [x[i] < x[j], x[i] + x[j] == t, abs(x[i] - x[j]) <= abs(t), (x[i] + x[j]) % m == r][qid]
                out += int(ok)
        return out
    if family_id == 6:
        y = [0] * n
        acc = 0
        for i in range(n):
            if eval_pred_vec(params["pid"], x, i, t, params):
                acc += eval_expr_vec(params["eid"], x, i, t, params)
            y[i] = acc
        return y
    if family_id == 7:
        return sorted(x[:n], reverse=bool(params["direction"]))
    raise ValueError("INVALID_FAMILY")


def formal_sexpr(family_id: int, params: dict[str, int]) -> str:
    if family_id == 0:
        return f"(if {COND_SCALAR[params['cid']]} {EXPR_SCALAR[params['sid_true']]} {EXPR_SCALAR[params['sid_false']]})"
    if family_id == 1:
        return f"(map {EXPR_VEC_SEXPR[params['eid']]})"
    if family_id == 2:
        return f"(sum {PRED_VEC_SEXPR[params['pid']]} {EXPR_VEC_SEXPR[params['eid']]})"
    if family_id == 3:
        return f"(count {PRED_VEC_SEXPR[params['pid']]})"
    if family_id == 4:
        return f"(search {PRED_VEC_SEXPR[params['pid']]})"
    if family_id == 5:
        return f"(paircount qid={params['qid']})"
    if family_id == 6:
        return f"(scan {PRED_VEC_SEXPR[params['pid']]} {EXPR_VEC_SEXPR[params['eid']]})"
    if family_id == 7:
        return "(sort asc)" if params["direction"] == 0 else "(sort desc)"
    if family_id == 8:
        return "(fib2 n)"
    raise ValueError("INVALID_FAMILY")


def signature_for_family(family_id: int) -> str:
    return "f(a:int,b:int,t:int)->int" if family_id == 0 else ("f(n:int,t:int)->int" if family_id == 8 else "f(x:list[int],n:int,t:int)->list[int]|int")


def serialize_params(params: dict[str, int]) -> str:
    return ",".join(f"{k}={params[k]}" for k in sorted(params))


def serialize_problem(index: int, family_id: int, params: dict[str, int], public_examples: list[tuple[tuple, Any]]) -> str:
    lines = [
        "<P>",
        f"I={index & MASK64:016x}",
        f"F={family_id}",
        f"S={signature_for_family(family_id)}",
        "D=I32,VI32,BOOL",
        f"K={serialize_params(params)}",
        f"X={formal_sexpr(family_id, params)}",
    ]
    for idx, (case, out) in enumerate(public_examples):
        lines.append(f"E{idx}={serialize_value(case)}->{serialize_value(out)}")
    lines.append("</P>")
    return "\n".join(lines) + "\n"


def state_line(**kwargs: Any) -> str:
    return ",".join(f"{k}={serialize_value(kwargs[k])}" for k in sorted(kwargs) if k in TRACE_VARS)


def trace_reference(family_id: int, params: dict[str, int], case: tuple, trace_max_lines: int) -> list[str]:
    if trace_max_lines == 0:
        return []
    lines: list[str] = []

    def add(**st: Any) -> bool:
        if len(lines) >= trace_max_lines:
            return False
        lines.append(f"@{len(lines):04d}|{state_line(**st)}")
        return True

    if family_id == 0:
        a, b, t = case
        add(a=a, b=b, t=t, r=run_reference_interpreter(family_id, params, case))
    elif family_id == 8:
        n, t = case
        a = b = 1
        add(a=a, b=b, i=0, n=n, t=t)
        i = 2
        while i <= n and add(a=a, b=b, i=i, n=n, t=t):
            a, b = b, a + b
            i += 1
    else:
        x, n, t = case
        y = [0] * n
        r0 = 0
        add(i=0, n=n, t=t, x=x, y=y, r=r0)
        for i in range(n):
            if family_id in (2, 3, 6) and eval_pred_vec(params["pid"], x, i, t, params):
                r0 += 1 if family_id == 3 else eval_expr_vec(params.get("eid", 0), x, i, t, params)
            if family_id == 6:
                y[i] = r0
            if not add(i=i, n=n, t=t, x=x, y=y, r=r0):
                break
    if len(lines) >= trace_max_lines:
        lines.append("@65535|T=1")
    return lines


def serialize_reasoning(family_id: int, params: dict[str, int], first_case: tuple, trace_max_lines: int) -> str:
    lines = ["<R>", f"A={formal_sexpr(family_id, params)}", f"I={serialize_value(first_case)}"]
    lines.extend(trace_reference(family_id, params, first_case, trace_max_lines))
    lines.append("</R>")
    return "\n".join(lines) + "\n"


def generate_record(stage: int, index: int) -> dict[str, Any]:
    if stage not in (1, 2, 3):
        raise ValueError("INVALID_STAGE")
    nonce = 0
    while True:
        rng = RNG(record_seed(stage, index, nonce))
        family_id = rng.weighted_choice_1000(FAMILY_WEIGHTS[stage])
        params = sample_params(family_id, rng)
        code = render_code(family_id, params)
        assert_ascii(code)
        if not ast_whitelist_ok(code):
            raise RuntimeError("RENDERER_EMITTED_INVALID_AST")
        fn = compile_restricted_function(code)
        public_cases = make_cases(family_id, stage, rng, PUBLIC_EXAMPLES)
        verify_cases = make_cases(family_id, stage, rng, 64)
        outputs_public = []
        for case in public_cases:
            y_oracle = run_reference_interpreter(family_id, params, case)
            if run_restricted(fn, case) != y_oracle:
                raise RuntimeError("REFERENCE_RENDER_MISMATCH")
            outputs_public.append((case, y_oracle))
        for case in verify_cases:
            if run_restricted(fn, case) != run_reference_interpreter(family_id, params, case):
                raise RuntimeError("REFERENCE_VERIFY_MISMATCH")
        P = serialize_problem(index, family_id, params, outputs_public)
        R = serialize_reasoning(family_id, params, public_cases[0], {1: 8, 2: 32, 3: 0}[stage])
        C = "<C>\n" + code + "</C>\n"
        train_text = ("<S1>\n" + R + C) if stage == 1 else (("<S2>\n" + P + C + R) if stage == 2 else ("<S3>\n" + P + C))
        token_ids = encode_ascii_record(train_text)
        if len(token_ids) <= 512:
            return {"stage": stage, "index": index, "family_id": family_id, "params": params, "P": P, "R": R, "C": C, "tests": verify_cases, "token_ids": token_ids}
        nonce += 1


def sha256_ascii(s: str) -> str:
    assert_ascii(s)
    return hashlib.sha256(s.encode("ascii")).hexdigest()


def extract_code_between_C_tags(out_ids: list[int] | bytes) -> str | None:
    raw = bytes([x for x in out_ids if 0 <= int(x) <= 255]) if not isinstance(out_ids, bytes) else out_ids
    text = raw.decode("ascii", errors="ignore")
    start = text.find("<C>\n")
    if start == -1:
        end = text.find("</C>\n")
        if end == -1:
            return None
        return text[:end] + "\n"
    end = text.find("</C>\n", start + 4)
    if end == -1:
        return None
    return text[start + 4:end] + "\n"


def verify_candidate_code(code: str, record: dict, tests_count: int) -> tuple[bool, str | None, str | None]:
    if not code.startswith("def f(") or not code.endswith("\n"):
        return False, None, None
    try:
        assert_ascii(code)
        if not ast_whitelist_ok(code):
            return False, None, None
        fn = compile_restricted_function(code)
    except Exception:
        return False, None, None
    rng = RNG(GLOBAL_SEED ^ 0xC0DEF00D ^ int(record["index"]))
    tests = make_cases(record["family_id"], 3, rng, tests_count)
    outputs = []
    for case in tests:
        try:
            y_candidate = run_restricted(fn, case)
            y_oracle = run_reference_interpreter(record["family_id"], record["params"], case)
        except Exception:
            return False, None, None
        if y_candidate != y_oracle:
            return False, None, None
        outputs.append(serialize_value(y_candidate))
    canonical_ast = ast.dump(ast.parse(code), annotate_fields=True, include_attributes=False)
    return True, sha256_ascii("|".join(outputs)), canonical_ast
