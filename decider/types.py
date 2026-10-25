from typing import List, Literal, NotRequired, TypedDict

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[
    str, JSONValue
]
type Content = str | dict[str, JSONValue]
type Label = int
type Target = float


class QuestionBase(TypedDict):
    instructions: NotRequired[Content | None]


class ChoiceQuestion(QuestionBase):
    type: Literal["choice"]
    criteria: dict[str, Content | None]


class ScoreQuestion(QuestionBase):
    type: Literal["score"]
    criteria: List[Content]


class NoulQuestion(QuestionBase):
    type: Literal["noul"]
    criteria: NotRequired[dict[Literal["true", "false"], Content | None] | None]


type Question = ChoiceQuestion | NoulQuestion | ScoreQuestion
type Questions = dict[str, Question]


class DecisionInput(TypedDict):
    state: Content
    questions: Questions


class Example(DecisionInput):
    label: Label
    target: Target
    source: dict[str, JSONValue]
