import ast
import tokenize
import keyword
from io import BytesIO
from pprint import pprint


class PyTokenizer:
    @staticmethod
    def get_lexical_tokens(code: bytes) -> list[tokenize.TokenInfo]:
        tokens = tokenize.tokenize(BytesIO(code).readline)
        return list(tokens)

    @staticmethod
    def get_exact_token_type(token: tokenize.TokenInfo):
        if token.type == tokenize.OP:
            return tokenize.EXACT_TOKEN_TYPES[token.string]
        return token.type

    @staticmethod
    def get_keywords():
        keywords: set[str] = set([*keyword.kwlist, *keyword.softkwlist])
        return list(keywords)

    @staticmethod
    def get_structure(code: bytes):
        parsed = ast.parse(code, mode="exec")
        return parsed

    @staticmethod
    def walk_ast(node: ast.AST, path: tuple = ()):
        node_type = type(node).__name__
        current_path = (*path, node_type)

        yield node, current_path

        for field_name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                yield from PyTokenizer.walk_ast(
                    value,
                    (*current_path, field_name),
                )
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        yield from PyTokenizer.walk_ast(
                            item,
                            (
                                *current_path,
                                field_name,
                                index,
                            ),
                        )

    @staticmethod
    def get_ast_roles(
        tree: ast.AST,
    ) -> dict[tuple[int, int], str]:
        roles = {}

        def walk(node: ast.AST, ast_role: str | None = None):
            if hasattr(node, "lineno") and hasattr(node, "col_offset"):
                start = (node.lineno, node.col_offset) # type: ignore
                roles[start] = {
                    "ast_role": ast_role,
                    "ast_node": type(node).__name__,
                }

            for field_name, value in ast.iter_fields(node):
                if isinstance(value, ast.AST):
                    walk(value, field_name)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            walk(item, field_name)

        walk(tree)
        return roles

    @staticmethod
    def get_ast_metadata(
        tree: ast.AST,
    ) -> list[dict]:
        metadata = []
        for node, path in PyTokenizer.walk_ast(tree):
            fields = {}
            for field_name, value in ast.iter_fields(node):
                if isinstance(value, ast.AST):
                    fields[field_name] = {
                        "type": "AST",
                        "value": type(value).__name__,
                    }
                elif isinstance(value, list):
                    fields[field_name] = [
                        (type(item).__name__ if isinstance(item, ast.AST) else item)
                        for item in value
                    ]
                else:
                    fields[field_name] = value

            metadata.append(
                {
                    "node_type": type(node).__name__,
                    "path": path,
                    "fields": fields,
                }
            )

        return metadata

    @staticmethod
    def get_metadata(code: bytes) -> dict:
        lexical_tokens = PyTokenizer.get_lexical_tokens(code)
        tree = PyTokenizer.get_structure(code)
        keywords = PyTokenizer.get_keywords()
        ast_metadata = PyTokenizer.get_ast_metadata(tree)

        tokens = []

        for token in lexical_tokens:
            exact_type = PyTokenizer.get_exact_token_type(token)
            token_metadata = {
                "text": token.string,
                "lexical": {
                    "type": tokenize.tok_name[token.type],
                    "exact_type": tokenize.tok_name[exact_type],
                    "start": token.start,
                    "end": token.end,
                },
            }
            if token.string in keywords:
                token_metadata["lexical"]["keyword"] = True

            tokens.append(token_metadata)

        return {
            "tokens": tokens,
            "ast": ast_metadata,
        }

    @staticmethod
    def get_role_metadata(code: bytes) -> list[dict]:
        lexical_tokens = PyTokenizer.get_lexical_tokens(code)
        tree = PyTokenizer.get_structure(code)
        ast_roles = PyTokenizer.get_ast_roles(tree)

        source = code.decode("utf-8")
        lines = source.splitlines(keepends=True)

        def get_offset(position: tuple[int, int]) -> int:
            line, column = position
            return sum(len(value) for value in lines[: line - 1]) + column

        metadata = []
        previous_offset = 0

        for token in lexical_tokens:
            if token.type in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
            }:
                continue

            start = get_offset(token.start)
            end = get_offset(token.end)
            gap = source[previous_offset:start]
            if gap:
                metadata.append(
                    {
                        "text": gap,
                        "exact_type": "WHITESPACE",
                        "ast_role": None,
                    }
                )

            exact_type = PyTokenizer.get_exact_token_type(token)
            ast_role = ast_roles.get(token.start)
            metadata.append(
                {
                    "text": token.string,
                    "exact_type": tokenize.tok_name[exact_type],
                    "ast_role": ast_role,
                }
            )
            previous_offset = end

        return metadata


if __name__ == "__main__":
    code = b'''
import math
import numpy as np
import torch.nn as nn
from transformers import PretrainedModel, AutoTokenizer

__id__ = "table"

@staticmethod
def calculate_total (items: dict) -> float:
    a: int = 5
    b: int = 6
    x: float = a + b
    y: float = 5.0
    y /= -b ** a
    y *= x + a ** b
    print(f"Hello world {6:d}")
    
    x = W_g @ X.T
    p = np.array(x[:, :, -1 : 1].transpose(1, 2, 3))

    while True:
        i = 3.0
    
    for i in range(5):
        print(5 + i)

    return sum(item["price"] * item["quantity"] for item in items)

if __name__ == "__main__":
    """This is a test"""
    """
    Hello
    World
    Input:
        - x 
    Returns:
        - x + 5
    """
    # Log the output
    print(calculate_total(items= [{"price": 10, "quantity": 12}]))
'''
    pytokenizer = PyTokenizer()
    tokens = pytokenizer.get_lexical_tokens(code)
    print(f"Total tokens {len(tokens)}")
    for token in tokens:
        print(token, pytokenizer.get_exact_token_type(token))

    # print("\n", sorted(pytokenizer.get_keywords()))

    # print(f"\n", ast.dump(pytokenizer.get_structure(code), indent=4))

    # tree = PyTokenizer.get_structure(code)

    # for node in PyTokenizer.walk_ast(tree):
    #    print(type(node).__annotations__)
    #    code = b"""
    # import math
    # def calculate_total(items: dict) -> float:
    #    a: int = 5
    #    b: int = 6
    #    x: float = a + b
    #    return x
    # """

    #metadata = PyTokenizer.get_role_metadata(code)
    #pprint(metadata)
