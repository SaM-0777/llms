import json
from pathlib import Path


def main():
    file_path = Path("tokenizer/notation_lexicals_II.json")
    with file_path.open("r", encoding="utf-8") as file:
        data: dict = json.load(file)

    mathematics = {token for token in data["mathematics"]["tokens"]}
    scientific_notation = {token for token in data["scientific_notation"]["tokens"]}
    unified = {
        token for language in data["languages"] for token in language.get("tokens", [])
    }

    #print(len(mathematics))
    #print(len(scientific_notation))
    print(len(unified))

    #unified.update(mathematics)
    #unified.update(scientific_notation)

    #print(len(unified))

    data["unified"] = list(unified)

    file_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nCollection complete")


if __name__ == "__main__":
    main()
