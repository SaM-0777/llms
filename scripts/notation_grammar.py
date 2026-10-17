import json
import re
import unicodedata
from pathlib import Path

import requests

OUTPUT_FILE = Path("notation_lexicals.json")

UNICODE_BASE = "https://www.unicode.org/Public/UCD/latest/ucd/"
UNICODE_DATA_URL = UNICODE_BASE + "UnicodeData.txt"
UNICODE_PROP_URL = UNICODE_BASE + "PropList.txt"
UNICODE_SCRIPTS_URL = UNICODE_BASE + "Scripts.txt"

LATEX_SYMLIST_URL = "https://tug.ctan.org/info/symbols/comprehensive/SYMLIST"

BIPM_REPOSITORY_API = (
    "https://api.github.com/repos/metanorma/"
    "bipm-si-brochure/git/trees/main?recursive=1"
)

BIPM_RAW_BASE = "https://raw.githubusercontent.com/metanorma/" "bipm-si-brochure/main/"


def download_text(url: str) -> str:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def parse_codepoint_range(value: str) -> tuple[int, int]:
    parts = value.strip().split("..")

    start = int(parts[0], 16)
    end = int(parts[-1], 16)

    return start, end


def collect_unicode_mathematics() -> set[str]:
    """
    Collect mathematical Unicode characters from the official UCD.

    Sources:
      - UnicodeData.txt: General_Category=Sm
      - PropList.txt: Other_Math
      - Scripts.txt: Greek letters

    No character lists are manually enumerated.
    """

    unicode_data = download_text(UNICODE_DATA_URL)
    prop_list = download_text(UNICODE_PROP_URL)
    scripts = download_text(UNICODE_SCRIPTS_URL)

    codepoints = set()
    greek_ranges = []

    # Mathematical symbols from the Unicode general category Sm.
    for line in unicode_data.splitlines():
        fields = line.split(";")

        if len(fields) < 3:
            continue

        codepoint = int(fields[0], 16)
        category = fields[2]

        if category == "Sm":
            codepoints.add(codepoint)

    # Additional mathematical characters from Unicode's Other_Math
    # binary property.
    for line in prop_list.splitlines():
        line = line.split("#", 1)[0].strip()

        if not line:
            continue

        fields = [field.strip() for field in line.split(";")]

        if len(fields) < 2 or fields[1] != "Other_Math":
            continue

        start, end = parse_codepoint_range(fields[0])
        codepoints.update(range(start, end + 1))

    # Greek letters are used extensively in mathematical notation.
    for line in scripts.splitlines():
        line = line.split("#", 1)[0].strip()

        if not line:
            continue

        fields = [field.strip() for field in line.split(";")]

        if len(fields) < 2 or fields[1] != "Greek":
            continue

        start, end = parse_codepoint_range(fields[0])
        greek_ranges.append((start, end))

    # Include assigned Greek letters, rather than every codepoint
    # in the Greek script ranges.
    for line in unicode_data.splitlines():
        fields = line.split(";")

        if len(fields) < 3:
            continue

        codepoint = int(fields[0], 16)
        category = fields[2]

        if category not in {"Lu", "Ll", "Lt", "Lm"}:
            continue

        if any(start <= codepoint <= end for start, end in greek_ranges):
            codepoints.add(codepoint)

    return {
        chr(codepoint)
        for codepoint in codepoints
        if unicodedata.category(chr(codepoint)) != "Cn"
    }


def collect_latex_commands() -> set[str]:
    """
    Extract LaTeX control sequences from CTAN's SYMLIST.

    The first column identifies the character or control sequence
    usable in a LaTeX document. Internal implementation commands in
    the second column are not treated as user-facing commands.
    """

    content = download_text(LATEX_SYMLIST_URL)
    commands = set()

    for line in content.splitlines():
        columns = re.split(r"\s{2,}", line.strip(), maxsplit=1)

        if not columns:
            continue

        usable_form = columns[0]

        commands.update(
            re.findall(
                r"\\(?:[A-Za-z@]+|[^A-Za-z\s])",
                usable_form,
            )
        )

    if not commands:
        raise RuntimeError("No LaTeX commands were extracted from CTAN SYMLIST.")

    return commands


def collect_bipm_scientific_notation() -> tuple[set[str], str]:
    """
    Extract SI unit identifiers and unit expressions from the
    structured BIPM SI Brochure source repository.

    This is specifically an SI-related collection, not a claim to
    cover every scientific notation convention.
    """

    response = requests.get(BIPM_REPOSITORY_API, timeout=60)
    response.raise_for_status()

    tree = response.json().get("tree", [])

    # Discover the available brochure version directories rather
    # than hardcoding a version number.
    version_paths = set()

    for item in tree:
        path = item.get("path", "")

        match = re.match(
            r"sources/si-brochure/([^/]+)/sections-en/[^/]+\.adoc$",
            path,
        )

        if match:
            version_paths.add(match.group(1))

    if not version_paths:
        raise RuntimeError(
            "No English SI Brochure source files were found "
            "in the BIPM source repository."
        )

    def version_key(version: str):
        return tuple(
            int(part) if part.isdigit() else part
            for part in re.split(r"[._-]", version)
        )

    version = max(version_paths, key=version_key)

    prefix = f"sources/si-brochure/{version}/sections-en/"

    source_files = [
        item["path"]
        for item in tree
        if item.get("type") == "blob"
        and item.get("path", "").startswith(prefix)
        and item["path"].endswith(".adoc")
    ]

    tokens = set()

    for path in source_files:
        content = download_text(BIPM_RAW_BASE + path)

        # The source represents SI symbols and expressions through
        # unitsml(...) notation.
        for match in re.finditer(r"unitsml\(([^)]*)\)", content):
            expression = match.group(1).strip()

            if expression:
                tokens.add(expression)

    if not tokens:
        raise RuntimeError(
            "No SI notation identifiers were extracted from "
            f"the BIPM source version {version}."
        )

    source = "BIPM SI Brochure source repository; " f"version directory {version}"

    return tokens, source


def load_existing_data() -> dict:
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"{OUTPUT_FILE} was not found. "
            "Run the programming-language collector first."
        )

    data = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))

    if not isinstance(data, dict) or "languages" not in data:
        raise ValueError("Expected the existing JSON to contain a 'languages' key.")

    return data


def main():
    data = load_existing_data()

    print("Collecting Unicode mathematical characters...")
    mathematics = collect_unicode_mathematics()

    print("Collecting LaTeX commands from CTAN...")
    latex = collect_latex_commands()

    print("Collecting SI notation from BIPM sources...")
    scientific, scientific_source = collect_bipm_scientific_notation()

    data["mathematics"] = {
        "source": [
            UNICODE_DATA_URL,
            UNICODE_PROP_URL,
            UNICODE_SCRIPTS_URL,
        ],
        "tokens": sorted(mathematics),
    }

    data["latex"] = {
        "source": LATEX_SYMLIST_URL,
        "tokens": sorted(latex),
    }

    data["scientific_notation"] = {
        "source": scientific_source,
        "tokens": sorted(scientific),
    }

    # Deduplicate exact strings across all collected domains.
    unified = {
        token for language in data["languages"] for token in language.get("tokens", [])
    }

    unified.update(mathematics)
    unified.update(latex)
    unified.update(scientific)

    data["unified"] = sorted(unified)

    OUTPUT_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nCollection complete")
    print(f"Mathematical characters: {len(mathematics)}")
    print(f"LaTeX commands: {len(latex)}")
    print(f"SI notation entries: {len(scientific)}")
    print(f"Unified unique strings: {len(unified)}")
    print(f"Output: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
