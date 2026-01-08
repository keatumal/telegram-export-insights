import json
import re
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from src.converter import messages_to_dataframe
from src.parser import parse_message

st.set_page_config(
    page_title="Telegram Chat Insights",
    page_icon="💬",
    layout="wide",
)

# --- Header & intro ---

st.title("Telegram Chat Insights")
st.markdown(
    "Upload a Telegram JSON export (`result.json`) and explore your chat activity."
)

# --- File upload ---

uploaded_file = st.file_uploader(
    "Upload Telegram JSON export",
    type=["json"],
)

if not uploaded_file:
    st.info("Drop a JSON export file here to get started.")
    st.stop()

# --- Parse JSON into DataFrame ---

data = json.load(uploaded_file)
messages = [parse_message(m) for m in data["messages"]]
df = messages_to_dataframe(messages)

# --- Date range selector ---

min_date = df["datetime"].min().date()
max_date = df["datetime"].max().date()

st.subheader("Date range")
dates = st.date_input(
    "Filter messages by date",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(dates) == 2:
    date_from, date_to = dates
else:
    date_from, date_to = dates[0], max_date

if date_from > date_to:
    st.error("Start date must be before end date.")
    st.stop()

# Filter df based on dates and save the original
mask = (df["datetime"].dt.date >= date_from) & (df["datetime"].dt.date <= date_to)
df_filtered = df[mask]

st.caption(
    f"Showing **{len(df_filtered)}** messages from **{date_from}** to **{date_to}**."
)

# Additional fields for analytics
df_filtered["weekday"] = df_filtered["datetime"].dt.day_name()
df_filtered["hour"] = df_filtered["datetime"].dt.hour

days_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

overview_tab, users_tab, reactions_tab = st.tabs(["Overview", "Users", "Reactions"])

# --- Overview tab ---

with overview_tab:

    # --- Messages per day ---

    st.subheader("Messages per day")
    daily_counts = (
        df_filtered.groupby(df_filtered["datetime"].dt.date)["id"]
        .count()
        .reset_index()
        .rename(columns={"datetime": "date", "id": "count"})
    )

    if daily_counts.empty:
        st.warning("No messages in the selected date range.")
    else:
        fig = px.line(
            daily_counts,
            x="date",
            y="count",
            markers=True,
            labels={"date": "Date", "count": "Messages"},
            title=None,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

    # --- Activity by hour ---

    st.subheader("Activity by hour")
    hourly = df_filtered.groupby(df_filtered["datetime"].dt.hour)["id"].count()
    fig = px.bar(
        hourly.reset_index(),
        x="datetime",
        y="id",
        title="Activity by hour",
        labels={"datetime": "Hour", "id": "Messages"},
    )
    fig.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=1), showlegend=False)
    st.plotly_chart(fig, width="stretch")

    # --- Activity by weekday ---

    st.subheader("Activity by weekday")
    weekly = df_filtered.groupby("weekday")["id"].count()
    weekly = weekly.reindex(days_order[::-1])

    fig = px.bar(
        weekly.reset_index(),
        x="id",
        y="weekday",
        orientation="h",
        labels={"id": "Messages", "weekday": "Day of week"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    # --- Activity by weekday × hour ---

    st.subheader("Activity by weekday × hour")
    heatmap_data = df_filtered.groupby(["weekday", "hour"])["id"].count().reset_index()
    heatmap_pivot = heatmap_data.pivot(
        index="weekday", columns="hour", values="id"
    ).fillna(0)
    heatmap_pivot = heatmap_pivot.reindex(days_order[::-1])

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale="YlOrRd",
        )
    )

    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Weekday",
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
    )
    st.plotly_chart(fig, width="stretch")

    # --- Content type ---

    st.subheader("Distribution by content type")
    message_df = df_filtered[df_filtered["type"] == "message"]

    content_types = {
        "Photo": message_df["has_photo"].sum(),
        "File": message_df["has_file"].sum(),
        "Voice": (message_df["media_type"] == "voice_message").sum(),
        "Video": (message_df["media_type"] == "video_message").sum(),
        "Poll": message_df["has_poll"].sum(),
        "Text": (
            (message_df["text"].str.len() > 0)
            & (~message_df["has_photo"])
            & (~message_df["has_file"])
            & (message_df["media_type"].isna())
        ).sum(),
    }

    content_types = {k: v for k, v in content_types.items() if v > 0}

    fig = px.bar(
        x=list(content_types.values()),
        y=list(content_types.keys()),
        orientation="h",
        labels={"x": "Quantity", "y": "Type"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    # --- Word cloud ---

    st.subheader("Most popular words")
    min_word_len = st.number_input(
        "Minimun word length",
        min_value=1,
        max_value=50,
        value=7,
        step=1,
    )

    all_text = " ".join(
        df_filtered[df_filtered["text"].notna()]["text"].astype(str).str.lower()
    )

    tokens = re.findall(r"\w+", all_text)
    tokens = [w for w in tokens if len(w) >= min_word_len]

    filtered_text = " ".join(tokens)

    if len(filtered_text) > 100:
        wordcloud = WordCloud(
            width=1200,
            height=800,
            background_color="white",
            min_word_length=min_word_len,
        ).generate(filtered_text)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig, width="stretch")
    else:
        st.info("Not enough text data for word cloud")


with users_tab:

    # --- Top-N selector ---

    users_top_n = st.number_input(
        "Number of top users to show",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )

    # --- Number of messages ---

    st.subheader(f"Top-{users_top_n} users by number of messages")
    top_msg_count = (
        df_filtered[df_filtered["type"] == "message"]
        .groupby("from")["id"]
        .count()
        .sort_values(ascending=False)
        .head(users_top_n)
    )

    fig = px.bar(
        top_msg_count.sort_values(ascending=True),
        orientation="h",
        labels={"value": "Messages", "from": "User"},
    )
    fig.update_layout(showlegend=False, height=max(400, users_top_n * 30))
    st.plotly_chart(fig, width="stretch")

    # --- Number of chars ---

    st.subheader(f"Top-{users_top_n} users by number of chars in messages")
    df_filtered["text_length"] = df_filtered["text"].str.len()
    avg_length = (
        df_filtered[df_filtered["text_length"] > 0]
        .groupby("from")["text_length"]
        .mean()
        .sort_values(ascending=False)
        .head(users_top_n)
    )

    fig = px.bar(
        avg_length.sort_values(ascending=True),
        orientation="h",
        labels={"value": "Chars", "from": "User"},
    )
    fig.update_layout(showlegend=False, height=max(400, users_top_n * 30))
    st.plotly_chart(fig, width="stretch")

    # -- Fastest responders ---

    st.subheader(f"Top-{users_top_n} fastest responders to messages")
    min_replies_num = st.number_input(
        "Min number of replies",
        min_value=1,
        max_value=100,
        value=10,
        step=5,
    )

    replies = df_filtered[df_filtered["is_reply"] == True].copy()
    msg_times = df_filtered.set_index("id")["datetime"].to_dict()

    delays = []
    for _, reply in replies.iterrows():
        original_id = reply["reply_to_message_id"]
        if original_id in msg_times:
            delay = (reply["datetime"] - msg_times[original_id]).total_seconds()
            if delay > 0:
                delays.append({"from": reply["from"], "delay_seconds": delay})

    delays_df = pd.DataFrame(delays)

    median_delays = delays_df.groupby("from")["delay_seconds"].agg(["median", "count"])
    median_delays = median_delays[median_delays["count"] >= min_replies_num]
    median_delays["median_minutes"] = median_delays["median"] / 60

    top_fast = median_delays.nsmallest(users_top_n, "median").sort_values(
        "median", ascending=False
    )

    fig = px.bar(
        top_fast,
        x="median_minutes",
        y=top_fast.index,
        orientation="h",
        labels={"median_minutes": "Median delay (minutes)", "from": "User"},
        text="median_minutes",
        color="median_minutes",
        color_continuous_scale="RdYlGn_r",
    )

    fig.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
    fig.update_layout(showlegend=False, height=max(400, users_top_n * 30))
    st.plotly_chart(fig, width="stretch")

with reactions_tab:

    # --- Top-N selector

    reactions_top_n = st.number_input(
        "Number of top users or reactions to show",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )

    # --- Top reactions ---

    st.subheader(f"Top-{reactions_top_n} reactions")
    reaction_stats = {}
    for msg in messages:
        for r in msg.reactions:
            reaction_stats[r.emoji] = reaction_stats.get(r.emoji, 0) + r.count

    sorted_reactions = sorted(reaction_stats.items(), key=lambda x: x[1], reverse=True)

    top_emojis = [e for e, _ in sorted_reactions[:reactions_top_n]]
    top_counts = [c for _, c in sorted_reactions[:reactions_top_n]]

    fig = px.bar(
        x=top_counts[::-1],
        y=top_emojis[::-1],
        orientation="h",
        labels={"x": "Quantity", "y": "Reaction"},
    )
    fig.update_layout(
        showlegend=False,
        height=max(400, reactions_top_n * 40),
        yaxis=dict(tickfont=dict(size=20)),
    )
    st.plotly_chart(fig, width="stretch")

    # --- Reaction lovers ---

    st.subheader(f"Top-{reactions_top_n} reaction lovers")
    # user -> emoji -> count
    user_emoji = defaultdict(lambda: defaultdict(int))

    for msg in messages:
        for r in msg.reactions:
            for recent in r.recent:
                user = recent.get("from")
                if user:
                    user_emoji[user][r.emoji] += 1

    rows = []
    for user, emojis in user_emoji.items():
        for emoji, count in emojis.items():
            rows.append({"user": user, "emoji": emoji, "count": count})

    emoji_df = pd.DataFrame(rows)

    user_totals = emoji_df.groupby("user")["count"].sum().sort_values(ascending=False)
    top_users = user_totals.head(reactions_top_n).index

    top_emoji_lovers = emoji_df[emoji_df["user"].isin(top_users)]

    fig = px.bar(
        top_emoji_lovers,
        x="count",
        y="user",
        color="emoji",
        orientation="h",
        labels={"count": "Reactions", "user": "User", "emoji": "Emoji"},
        category_orders={"user": top_users.tolist()},
    )
    fig.update_layout(
        showlegend=False,
        height=max(400, reactions_top_n * 35),
    )
    st.plotly_chart(fig, width="stretch")

    # --- Favorite reactions ---

    st.subheader("Favorite reactions")
    fav_emoji = (
        emoji_df.sort_values(["user", "count"], ascending=[True, False])
        .groupby("user")
        .head(1)
        .sort_values("count", ascending=False)
    )

    fig = px.bar(
        fav_emoji.head(reactions_top_n).sort_values(by="count", ascending=True),
        x="count",
        y="user",
        orientation="h",
        text="emoji",
        labels={"count": "Reactions", "user": "User"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        height=max(400, reactions_top_n * 35),
    )
    st.plotly_chart(fig, width="stretch")
