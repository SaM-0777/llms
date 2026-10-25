from dataclasses import dataclass, field as dataclass_field
import json
import math
import random
from typing import Any, Counter


def parse(value: Any):
    try:
        return json.loads(value)
    except:
        return value


def build_options(kind, criteria) -> str:
    options = []

    if isinstance(criteria, dict):
        for option_id, description in criteria.items():
            option = {"option_id": option_id, "description": description}
            options.append(json.dumps(option))
    elif isinstance(criteria, list):
        for option_id, description in enumerate(criteria):
            option = {"option_id": option_id, "description": description}
            options.append(option)
    else:
        raise ValueError(f"Criteria must be a valid dict or list")

    return "".join([f"\nOPTION {i + 1}: {option}" for i, option in enumerate(options)])


def format_message(row):
    state = parse(row["state"])
    questions = parse(row["questions"])
    gold = parse(row["gold"])

    fields = []

    if not state or not isinstance(questions, dict) or not isinstance(gold, dict):
        return None

    for idx, (name, question) in enumerate(questions.items()):
        if name not in gold:
            continue

        kind, instructions = question.get("type"), question.get("instructions")
        if not kind or not instructions:
            continue

        criteria = question["criteria"]
        options = build_options(kind, criteria)

        field = f"\n\nFIELD {idx + 1}\nID: {name}\nTYPE: {kind}\nINSTRUCTION: {instructions}\nALLOWED_OPTIONS: {options}\nEND FIELD"

        fields.append(field)

    prompt = f"\nSTATE: {state}\n\nSCHEMA FIELDS: \n\n{"\n\n".join(fields)}"

    return prompt


def format_question(question):
    if not isinstance(question, dict) or not question.get("type"):
        raise ValueError("not a valid question")
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        if (
            isinstance(criteria, list)
            and criteria
            and all(isinstance(c, str) for c in criteria)
        ):
            criteria = dict.fromkeys(criteria)
        if not (isinstance(criteria, dict) and criteria):
            raise ValueError("needs criteria naming its options")
        if len({str(key) for key in criteria}) != len(criteria):
            raise ValueError("has repeated options")
    if kind == "score" and not (isinstance(criteria, list) and criteria):
        raise ValueError("needs a list of criteria levels")
    if kind == "noul" and criteria is not None:
        if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
            raise ValueError('criteria may only have "true" and "false"')
    formatted_question = {"type": kind, "instructions": question.get("instructions")}
    if criteria is not None:
        formatted_question["criteria"] = criteria
    return formatted_question


def question_options(question) -> list[tuple[str, str]]:
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


def _label(kind: str, label):
    if (
        kind == "score"
        and isinstance(label, (int, float, str))
        and not isinstance(label, bool)
    ):
        try:
            level = float(label)
        except ValueError:
            return None
        return str(int(level)) if level.is_integer() else None
    if label is None:
        return None
    return str(label).strip().lower() if kind == "noul" else str(label)


def target_for(kind, keys, gold):
    if not isinstance(gold, dict):
        gold = {"label": gold}
    label = _label(kind, gold.get("label"))
    probabilities = gold.get("probabilities")
    if kind == "noul" and not isinstance(probabilities, dict):
        noul = gold.get("noul")
        probabilities = {"true": noul} if isinstance(noul, (int, float)) else None
    if isinstance(probabilities, dict):
        try:
            values = {str(key): float(value) for key, value in probabilities.items()}
        except (TypeError, ValueError):
            values = {}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("gold has probabilities that are not finite numbers")
        if kind == "noul" and len(values.keys() & {"false", "true"}) == 1:
            known = "true" if "true" in values else "false"
            values[{"true": "false", "false": "true"}[known]] = 1.0 - values[known]
        target = [max(0.0, values.get(key, 0.0)) for key in keys]
        total = sum(target)
        if 0 < total < math.inf:
            target = [value / total for value in target]
            return target, (
                keys.index(label) if label in keys else target.index(max(target))
            )
    if label in keys:
        return [1.0 if key == label else 0.0 for key in keys], keys.index(label)
    raise ValueError("gold has no usable label or probabilities")


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    record_id: str
    media: dict[str, Any] | None = dataclass_field(
        default=None, compare=False, repr=False
    )


def _tokens(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
# CLEF_TYPES = ("choice", "score", "noul")


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int = 16384,
    max_state_tokens: int | None = None,
    processor: Any | None = None,
) -> EncodedRecord:
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions: list[EncodedQuestion] = []
    for question_index, (question_id, question) in enumerate(
        record["questions"].items()
    ):
        schema_ids.extend(
            _tokens(
                tokenizer,
                f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: ",
            )
        )
        question_start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        question_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))

        option_spans: list[tuple[int, int]] = []
        option_ids: list[str] = []
        for option_index, (option_id, description) in enumerate(
            question_options(question)
        ):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        questions.append(
            EncodedQuestion(
                question_id=str(question_id),
                question_type=QUESTION_TYPES[str(question["type"])],
                question_span=(question_start, question_end),
                option_spans=tuple(option_spans),
                option_ids=tuple(option_ids),
            )
        )

    prefix_ids = _tokens(
        tokenizer,
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n",
    )
    suffix_ids = _tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )
    # media_ids, media = _encode_media(processor, record)
    # if media is not None:
    #    media["token_offset"] = len(prefix_ids)
    #    prefix_ids = prefix_ids + media_ids
    state_ids = _tokens(tokenizer, render(record["state"]))
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise ValueError(
            f"schema requires {fixed_length} tokens before state; maximum is {max_length}"
        )
    state_ids = state_ids[: max_length - fixed_length]
    schema_offset = len(prefix_ids) + len(state_ids)
    shifted_questions = tuple(
        EncodedQuestion(
            question_id=question.question_id,
            question_type=question.question_type,
            question_span=(
                question.question_span[0] + schema_offset,
                question.question_span[1] + schema_offset,
            ),
            option_spans=tuple(
                (start + schema_offset, end + schema_offset)
                for start, end in question.option_spans
            ),
            option_ids=question.option_ids,
        )
        for question in questions
    )
    input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
    if not input_ids or not shifted_questions:
        raise ValueError("record produced no model input or questions")
    return EncodedRecord(
        input_ids=input_ids,
        questions=shifted_questions,
        record_id=str(record.get("id", "unknown")),
        # media=media,
    )


def build_dataset(rows, tokenizer, max_length=2048):
    total = 0
    skipped = 0
    truncated = 0
    items = []

    for index, row in enumerate(rows):
        row = row if isinstance(row, dict) else {}
        state = parse(row["state"])
        questions = parse(row["questions"])
        gold = parse(row["gold"])

        if state is None or not isinstance(state, dict) or not isinstance(gold, dict):
            skipped += 1
            total += 1
            continue

        kept, targets, labels = {}, [], []
        for name, question in questions.items():
            total += 1

            if name not in gold:
                continue

            try:
                formatted_question = format_question(question)
                keys = [key for key, _ in question_options(formatted_question)]
                target, label = target_for(
                    formatted_question["type"], keys, parse(gold[name])
                )
            except (TypeError, ValueError, KeyError) as exc:
                skipped += 1
                continue

            kept[str(name)], targets, labels = (
                formatted_question,
                targets + [target],
                labels + [label],
            )

        if not kept:
            continue

        source = {"state": state, "questions": kept}
        try:
            record = encode_record(tokenizer, source, max_length, processor=None)
        except (TypeError, ValueError, OSError) as exc:
            for name in kept:
                skipped += 1
            continue

        items.append(
            {
                "input_ids": list(record.input_ids),
                "record": record,
                "qtypes": [QUESTION_TYPES[q["type"]] for q in kept.values()],
                "targets": targets,
                "labels": labels,
                "row": index,
                "source": source,
            }
        )

    return items, total, skipped


@staticmethod
def split_holdout(
    items: list,
    seed: int = 3407,
    fraction: float = 0.1,
    max_items: int = 400,
):
    # Counted in decisions: a Clef item holds every question of its row.
    sizes = Counter()
    for item in items:
        sizes[item["row"]] += len(item.get("labels", (None,)))
    target = min(max_items, int(sum(sizes.values()) * fraction))
    rows = sorted(sizes)
    random.Random(seed).shuffle(rows)
    held, count = set(), 0
    # Whole rows that still fit under the target; the last row always stays in training.
    for row in rows[:-1]:
        if count + sizes[row] <= target:
            held.add(row)
            count += sizes[row]
    return (
        [item for item in items if item["row"] not in held],
        [item for item in items if item["row"] in held],
    )
