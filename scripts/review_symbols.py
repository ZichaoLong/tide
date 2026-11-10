"""Syntax-backed symbol extraction for an implementation audit.

Python uses the standard AST. C++ uses an explicitly requested Clang invocation;
failed or unavailable parsing stays visible, including any partial declarations.
Neither parser establishes that a function has been reviewed or is correct.
"""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import tempfile


def digest(value):
    return hashlib.sha256(value).hexdigest()


def python_symbols(raw, filename):
    try:
        tree = ast.parse(raw, filename=filename)
    except (SyntaxError, ValueError) as error:
        return [], dict(method="python-ast", state="failed", diagnostic=str(error))
    lines = raw.splitlines(keepends=True)
    symbols = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []

        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def function(self, node):
            name = getattr(node, "name", "<lambda>")
            start = min([node.lineno] + [x.lineno for x in getattr(node, "decorator_list", ())])
            qualified = ".".join(self.scope + [name])
            symbols.append(dict(id=f"{qualified}@{node.lineno}:{node.col_offset}",
                name=qualified, kind=type(node).__name__, line=start,
                end_line=node.end_lineno, column=node.col_offset,
                sha256=digest(b"".join(lines[start-1:node.end_lineno])), definition=True))
            self.scope.append(name)
            self.generic_visit(node)
            self.scope.pop()

        visit_FunctionDef = function
        visit_AsyncFunctionDef = function
        visit_Lambda = function

    Visitor().visit(tree)
    return symbols, dict(method="python-ast", state="parsed", parser_version=platform.python_version(),
        scope="all syntactic functions, async functions and lambdas; dynamic/generated callables excluded")


def _json_objects(raw):
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(raw):
        while offset < len(raw) and raw[offset].isspace():
            offset += 1
        if offset == len(raw):
            break
        value, offset = decoder.raw_decode(raw, offset)
        yield value


@lru_cache(maxsize=None)
def _compiler_version(executable):
    try:
        result = subprocess.run([executable, "--version"], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=5)
        return result.stdout.decode("utf-8", errors="replace")[:4000].strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return "unavailable: "+str(error)


def clang_symbols(raw, filename, root, *, driver="clang++", arguments=(), timeout=15,
                  max_output_bytes=64*1024**2):
    executable = shutil.which(driver)
    if executable is None:
        return [], dict(method="clang-json-ast", state="unavailable", diagnostic=f"missing {driver}")
    path = (Path(root)/filename).resolve()
    command = [executable, "-x", "c++", "-std=c++17", "-fsyntax-only", "-fno-color-diagnostics",
               "-Xclang", "-ast-dump=json", *arguments, str(path)]
    recorded_command = [*command[:-1], filename]

    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_output_bytes, max_output_bytes))

    with tempfile.TemporaryFile() as output:
        try:
            process = subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.PIPE,
                timeout=timeout, preexec_fn=limits)
        except subprocess.TimeoutExpired:
            return [], dict(method="clang-json-ast", state="timeout", timeout_seconds=timeout,
                            command=recorded_command, scope="no completeness claim")
        size = output.tell()
        output.seek(0)
        text = output.read().decode("utf-8", errors="replace")
    parse = dict(method="clang-json-ast", state="parsed" if process.returncode == 0 else "partial",
        exit_code=process.returncode, command=recorded_command, parser_version=_compiler_version(executable),
        diagnostic=process.stderr.decode("utf-8", errors="replace").replace(str(Path(root).resolve()), "<source>")[-8000:], ast_bytes=size,
        scope="configured translation unit; inactive preprocessor branches and generated functions excluded")
    if size >= max_output_bytes:
        parse.update(state="output_limit", scope="no completeness claim")
        return [], parse
    kinds = {"FunctionDecl", "CXXMethodDecl", "CXXConstructorDecl", "CXXDestructorDecl", "CXXConversionDecl"}
    contexts, symbols, seen = {}, [], set()
    current_file = str(path)

    def visit(node, scope=()):
        nonlocal current_file
        location = node.get("loc", {})
        location = location.get("expansionLoc", location)
        begin = node.get("range", {}).get("begin", {})
        end = node.get("range", {}).get("end", {})
        begin = begin.get("expansionLoc", begin)
        end = end.get("expansionLoc", end)
        explicit = location.get("file", begin.get("file"))
        if explicit:
            current_file = str(Path(explicit).resolve())
        origin = current_file
        kind, name = node.get("kind"), node.get("name", "")
        nested = scope
        if kind in {"NamespaceDecl", "CXXRecordDecl", "RecordDecl"} and name:
            nested = scope + (name,)
            contexts[node.get("id")] = nested
        if kind in kinds and not node.get("isImplicit") and origin == str(path):
            start, stop = begin.get("offset"), end.get("offset")
            if start is not None and stop is not None and 0 <= start <= stop < len(raw):
                stop += end.get("tokLen", 1)
                line = raw.count(b"\n", 0, start)+1
                last = raw.count(b"\n", 0, stop)+1
                qualified = "::".join(contexts.get(node.get("parentDeclContextId"), scope)+(name,))
                identity = f"{qualified}@{line}:{start}:"+node.get("type", {}).get("qualType", "")
                if identity not in seen:
                    seen.add(identity)
                    definition = any(x.get("kind") in {"CompoundStmt", "CXXTryStmt"} for x in node.get("inner", ()))
                    symbols.append(dict(id=identity, name=qualified, kind=kind, line=line, end_line=last,
                        sha256=digest(raw[start:stop]), definition=definition,
                        type=node.get("type", {}).get("qualType")))
        for child in node.get("inner", ()):
            visit(child, nested)

    try:
        for tree in _json_objects(text):
            visit(tree)
    except (ValueError, TypeError) as error:
        parse.update(state="invalid_ast", diagnostic=parse["diagnostic"]+"\n"+str(error))
        symbols = []
    return symbols, parse
