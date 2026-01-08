from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TextEntity:
    type: str
    text: str
    language: Optional[str] = None  # for pre
    href: Optional[str] = None  # for text_link
    collapsed: Optional[bool] = None  # for blockquote


@dataclass
class Reaction:
    emoji: str
    count: int
    recent: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PollAnswer:
    text: str
    voters: int
    chosen: bool


@dataclass
class Poll:
    question: str
    closed: bool
    total_voters: int
    answers: List[PollAnswer]


@dataclass
class Message:
    """Base class for all messages"""

    id: int
    type: str  # "message" или "service"
    date: str
    date_unixtime: int
    edited: Optional[str] = None
    edited_unixtime: Optional[int] = None
    text_entities: List[TextEntity] = field(default_factory=list)
    reactions: List[Reaction] = field(default_factory=list)

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(int(self.date_unixtime))

    @property
    def text_plain(self) -> str:
        """Extracts plain text from text_entities"""
        return "".join(e.text for e in self.text_entities)


@dataclass
class RegularMessage(Message):
    """Regular user message"""

    from_: str = ""
    from_id: str = ""
    reply_to_message_id: Optional[int] = None

    # Media
    photo: Optional[str] = None
    photo_file_size: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    # Files & Media
    file: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    media_type: Optional[str] = None  # voice_message, video_message
    mime_type: Optional[str] = None
    duration_seconds: Optional[int] = None

    # Poll
    poll: Optional[Poll] = None


@dataclass
class ServiceMessage(Message):
    """Service message"""

    actor: str = ""
    actor_id: str = ""
    action: str = ""
    title: Optional[str] = None
    members: List[str] = field(default_factory=list)
    inviter: Optional[str] = None
