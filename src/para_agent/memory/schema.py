from typing import Literal

from pydantic import Field

from para_agent.diagnose.schema import Diagnosis, DiagnosisPair
from para_agent.schema import Schema


class ErrorNote(Schema):
    turn: int = Field(ge=0)
    type: Literal["조사", "어미", "시제", "높임·문체", "어휘 선택", "맞춤법", "띄어쓰기",
                  "문장 구조", "연결 표현", "지시어", "기타"]
    example: str
    fix: str


class ProfilePatch(Schema):
    topic: str | None = None
    style_notes: str | None = None
    thesis: str | None = None


class MemoryUpdate(Schema):
    error_notes: list[ErrorNote]
    profile_update: ProfilePatch
    turn_summary: str


class Profile(ProfilePatch):
    prompt: str
    # Only the code classifier may populate this field; it is absent from LLM patches.
    register_: Literal['하십시오체','해요체','해라체','해체','기타'] | None = Field(default=None, alias='register')

    def apply(self, update: MemoryUpdate) -> None:
        for key, value in update.profile_update.model_dump(exclude_unset=True, by_alias=False).items():
            setattr(self, key, value)


class SessionMemory(Schema):
    turn_diags: list[dict] = Field(default_factory=list)
    error_notes: list[ErrorNote] = Field(default_factory=list)
    turn_summaries: list[str] = Field(default_factory=list)

    def add_turn_diag(self, turn: int, diag: Diagnosis | DiagnosisPair) -> None:
        if self.turn_diags and turn <= self.turn_diags[-1]["turn"]:
            raise ValueError("Diagnosis turns must increase")
        self.turn_diags.append({"turn": turn, **diag.model_dump()})

    def notes(self) -> list[dict]:
        return [note.model_dump() for note in self.error_notes]

    def apply(self, update: MemoryUpdate) -> None:
        self.error_notes.extend(update.error_notes)
        self.turn_summaries.append(update.turn_summary)

