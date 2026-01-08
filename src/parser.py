from typing import Dict, List, Optional

from src.models import (
    Message,
    Poll,
    PollAnswer,
    Reaction,
    RegularMessage,
    ServiceMessage,
    TextEntity,
)


def parse_text_entities(entities_data: List[Dict]) -> List[TextEntity]:
    if not entities_data:
        return []
    return [
        TextEntity(
            type=e["type"],
            text=e["text"],
            language=e.get("language"),
            href=e.get("href"),
            collapsed=e.get("collapsed"),
        )
        for e in entities_data
    ]


def parse_reactions(reactions_data: List[Dict]) -> List[Reaction]:
    if not reactions_data:
        return []
    return [
        Reaction(emoji=r.get("emoji", ""), count=r["count"], recent=r.get("recent", []))
        for r in reactions_data
    ]


def parse_poll(poll_data: Optional[Dict]) -> Optional[Poll]:
    if not poll_data:
        return None
    return Poll(
        question=poll_data["question"],
        closed=poll_data["closed"],
        total_voters=poll_data["total_voters"],
        answers=[PollAnswer(**a) for a in poll_data["answers"]],
    )


def parse_message(msg: Dict) -> Message:
    """Parse the JSON message into the corresponding dataclass"""
    base_fields = {
        "id": msg["id"],
        "type": msg["type"],
        "date": msg["date"],
        "date_unixtime": int(msg["date_unixtime"]),
        "edited": msg.get("edited"),
        "edited_unixtime": (
            int(msg["edited_unixtime"]) if msg.get("edited_unixtime") else None
        ),
        "text_entities": parse_text_entities(msg.get("text_entities", [])),
        "reactions": parse_reactions(msg.get("reactions", [])),
    }

    if msg["type"] == "service":
        return ServiceMessage(
            **base_fields,
            actor=msg["actor"],
            actor_id=msg["actor_id"],
            action=msg["action"],
            title=msg.get("title"),
            members=msg.get("members", []),
            inviter=msg.get("inviter"),
        )
    else:  # message
        return RegularMessage(
            **base_fields,
            from_=msg.get("from", ""),
            from_id=msg.get("from_id", ""),
            reply_to_message_id=msg.get("reply_to_message_id"),
            photo=msg.get("photo"),
            photo_file_size=msg.get("photo_file_size"),
            width=msg.get("width"),
            height=msg.get("height"),
            file=msg.get("file"),
            file_name=msg.get("file_name"),
            file_size=msg.get("file_size"),
            media_type=msg.get("media_type"),
            mime_type=msg.get("mime_type"),
            duration_seconds=msg.get("duration_seconds"),
            poll=parse_poll(msg.get("poll")),
        )
