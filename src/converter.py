from typing import Any, Dict, List, Optional

import pandas as pd

from src.parser import Message, RegularMessage, ServiceMessage


def messages_to_dataframe(messages: List[Message]) -> pd.DataFrame:
    """Converts a list of messages into a flat DataFrame"""
    rows = []
    for msg in messages:
        row = {
            "id": msg.id,
            "type": msg.type,
            "datetime": msg.datetime,
            "text": msg.text_plain,
            "edited": msg.edited_unixtime is not None,
            "reactions_count": len(msg.reactions),
            "from": None,
            "from_id": None,
            "is_reply": False,
            "reply_to_message_id": None,
            "has_photo": False,
            "has_file": False,
            "media_type": None,
            "has_poll": False,
            "actor": None,
            "action": None,
        }

        if isinstance(msg, RegularMessage):
            row.update(
                {
                    "from": msg.from_,
                    "from_id": msg.from_id,
                    "is_reply": msg.reply_to_message_id is not None,
                    "reply_to_message_id": msg.reply_to_message_id,
                    "has_photo": msg.photo is not None,
                    "has_file": msg.file is not None,
                    "media_type": msg.media_type,
                    "has_poll": msg.poll is not None,
                }
            )
        elif isinstance(msg, ServiceMessage):
            row.update({"actor": msg.actor, "action": msg.action})

        rows.append(row)

    return pd.DataFrame(rows)
