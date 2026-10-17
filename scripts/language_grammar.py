import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OUTPUT_FILE = Path("lexicals.json")

BRANCHES = ("master", "main")

# language: (GitHub repository, possible paths to node-types.json)
GRAMMARS = {
    "python": (
        "tree-sitter/tree-sitter-python",
        ["src/node-types.json"],
    ),
    "c": (
        "tree-sitter/tree-sitter-c",
        ["src/node-types.json"],
    ),
    "cpp": (
        "tree-sitter/tree-sitter-cpp",
        ["src/node-types.json"],
    ),
    "c_sharp": (
        "tree-sitter/tree-sitter-c-sharp",
        ["src/node-types.json"],
    ),
    "javascript": (
        "tree-sitter/tree-sitter-javascript",
        ["src/node-types.json"],
    ),
    "typescript": (
        "tree-sitter/tree-sitter-typescript",
        ["typescript/src/node-types.json"],
    ),
    "tsx": (
        "tree-sitter/tree-sitter-typescript",
        ["tsx/src/node-types.json"],
    ),
    "java": (
        "tree-sitter/tree-sitter-java",
        ["src/node-types.json"],
    ),
    "go": (
        "tree-sitter/tree-sitter-go",
        ["src/node-types.json"],
    ),
    "rust": (
        "tree-sitter/tree-sitter-rust",
        ["src/node-types.json"],
    ),
    "ruby": (
        "tree-sitter/tree-sitter-ruby",
        ["src/node-types.json"],
    ),
    "php": (
        "tree-sitter/tree-sitter-php",
        ["src/node-types.json"],
    ),
    "bash": (
        "tree-sitter/tree-sitter-bash",
        ["src/node-types.json"],
    ),
    "json": (
        "tree-sitter/tree-sitter-json",
        ["src/node-types.json"],
    ),
    "yaml": (
        "tree-sitter-grammars/tree-sitter-yaml",
        ["src/node-types.json"],
    ),
    "toml": (
        "tree-sitter-grammars/tree-sitter-toml",
        ["src/node-types.json"],
    ),
    "html": (
        "tree-sitter/tree-sitter-html",
        ["src/node-types.json"],
    ),
    "css": (
        "tree-sitter/tree-sitter-css",
        ["src/node-types.json"],
    ),
    "lua": (
        "tree-sitter/tree-sitter-lua",
        ["src/node-types.json"],
    ),
    "swift": (
        "tree-sitter/tree-sitter-swift",
        ["src/node-types.json"],
    ),
    "kotlin": (
        "fwcd/tree-sitter-kotlin",
        ["src/node-types.json"],
    ),
    "scala": (
        "tree-sitter/tree-sitter-scala",
        ["src/node-types.json"],
    ),
    "haskell": (
        "tree-sitter/tree-sitter-haskell",
        ["src/node-types.json"],
    ),
    "elixir": (
        "elixir-lang/tree-sitter-elixir",
        ["src/node-types.json"],
    ),
    "ocaml": (
        "tree-sitter/tree-sitter-ocaml",
        ["grammars/ocaml/src/node-types.json", "src/node-types.json"],
    ),
    "dart": (
        "UserNobody14/tree-sitter-dart",
        ["src/node-types.json"],
    ),
    "julia": (
        "tree-sitter/tree-sitter-julia",
        ["src/node-types.json"],
    ),
    "r": (
        "r-lib/tree-sitter-r",
        ["src/node-types.json"],
    ),
    "perl": (
        "tree-sitter-perl/tree-sitter-perl",
        ["src/node-types.json"],
    ),
    "sql": (
        "Derekstride/tree-sitter-sql",
        ["src/node-types.json"],
    ),
    "dockerfile": (
        "camdencheek/tree-sitter-dockerfile",
        ["src/node-types.json"],
    ),
    "make": (
        "tree-sitter-grammars/tree-sitter-make",
        ["src/node-types.json"],
    ),
    "regex": (
        "tree-sitter/tree-sitter-regex",
        ["src/node-types.json"],
    ),
    "markdown": (
        "tree-sitter-grammars/tree-sitter-markdown",
        [
            "tree-sitter-markdown/src/node-types.json",
            "tree-sitter-markdown-inline/src/node-types.json",
            "src/node-types.json",
        ],
    ),
}


def collect_language(
    language: str,
    repository: str,
    paths: list[str],
) -> dict | None:
    """
    Download a grammar's generated node-types.json and collect
    its anonymous literal terminals.
    """

    with requests.Session() as session:
        session.headers.update({"User-Agent": "programming-lexical-collector"})

        for branch in BRANCHES:
            for path in paths:
                url = (
                    f"https://raw.githubusercontent.com/"
                    f"{repository}/{branch}/{path}"
                )

                try:
                    response = session.get(url, timeout=20)

                    if response.status_code == 404:
                        continue

                    response.raise_for_status()
                    node_types = response.json()

                    if not isinstance(node_types, list):
                        continue

                    tokens = {
                        node["type"]
                        for node in node_types
                        if (
                            node.get("named") is False
                            and isinstance(node.get("type"), str)
                            and node["type"]
                        )
                    }

                    if not tokens:
                        continue

                    return {
                        "language": language,
                        "source": url,
                        "tokens": sorted(tokens),
                    }

                except (
                    requests.RequestException,
                    json.JSONDecodeError,
                ):
                    continue

    return None


def main():
    results = []
    failures = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                collect_language,
                language,
                repository,
                paths,
            ): language
            for language, (repository, paths) in GRAMMARS.items()
        }

        for future in as_completed(futures):
            language = futures[future]

            try:
                result = future.result()

                if result is None:
                    failures.append(language)
                    print(f"[FAILED] {language}")
                else:
                    results.append(result)
                    print(f"[OK] {language}: " f"{len(result['tokens'])} terminals")

            except Exception as error:
                failures.append(language)
                print(f"[ERROR] {language}: {error}")

    results.sort(key=lambda item: item["language"])

    # Combine tokens from every language and remove duplicates.
    unified_tokens = sorted(
        {token for language_data in results for token in language_data["tokens"]}
    )

    # Preserve language-specific tokens and add the unified collection.
    output = {
        "languages": results,
        "unified": unified_tokens,
    }

    OUTPUT_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nCollection complete")
    print(f"Languages configured: {len(GRAMMARS)}")
    print(f"Languages collected: {len(results)}")
    print(f"Languages failed: {len(failures)}")
    print(f"Unique tokens across languages: {len(unified_tokens)}")
    print(f"Output: {OUTPUT_FILE.resolve()}")

    if failures:
        print("\nFailed languages:")
        for language in sorted(failures):
            print(f"  - {language}")


if __name__ == "__main__":
    main()
