from typing import Literal

from pydantic import BaseModel


class ExplainedTest(BaseModel):
    test: str
    plain_english: str


class FlaggedTest(BaseModel):
    test: str
    severity: Literal["Normal", "Watch", "High priority", "Critical"]
    reason: str


class DoctorQuestion(BaseModel):
    question: str


class AgentOutputs(BaseModel):
    explanations: list[ExplainedTest]
    flags: list[FlaggedTest]
    questions: list[DoctorQuestion]
    disclaimer: str
