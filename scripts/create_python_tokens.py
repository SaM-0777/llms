import keyword
import tokenize as py_tokenize

OUTPUT_FILE = "python_tokens.txt"


def get_python_tokens():
    keywords = list(keyword.kwlist) + list(keyword.softkwlist)
    operators = list(py_tokenize.EXACT_TOKEN_TYPES.keys())
    return list(dict.fromkeys(keywords + operators))


def main():
    tokens = get_python_tokens()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for token in tokens:
            f.write(token + "\n")

    print(f"Saved {len(tokens)} tokens to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
