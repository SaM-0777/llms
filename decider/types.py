from typing import List, Literal, NotRequired, TypedDict, Union

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


class ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    confidence: float
    # choice: str
    label: str
    probabilities: dict[str, float]


class NoulAnswer(TypedDict):
    type: Literal["noul"]
    confidence: float
    label: str
    noul: float
    probabilities: dict[str, float]


class ScoreAnswer(TypedDict):
    type: Literal["score"]
    confidence: float
    # legend: dict[str, Content | None]
    label: str
    probabilities: dict[str, float]
    score: float


Answer = Union[ChoiceAnswer, NoulAnswer, ScoreAnswer]


class DecisionInput(TypedDict):
    state: Content
    questions: Questions
    gold: dict[str, Answer]


class Example(DecisionInput):
    label: Label
    target: Target
    source: dict[str, JSONValue]
