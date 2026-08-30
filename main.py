"""
Unhinged Waifu — a silly persona chatbot built on Streamlit + Groq.

Cleaned up from the original: dead code removed, persona data centralized,
config validated up front, and the request/response flow simplified.
"""

import os
import random
from dataclasses import dataclass, field

import streamlit as st
from openai import OpenAI

# -------------------------------------------------------------------
# Session state must be initialized before st.set_page_config() if the
# page config (title/icon) is going to depend on it.
# -------------------------------------------------------------------

if "normal_mode" not in st.session_state:
    st.session_state.normal_mode = True

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

if st.session_state.normal_mode:
    st.set_page_config(page_title="Chat Assistant", page_icon="💬")
else:
    st.set_page_config(page_title="Unhinged Waifu", page_icon="💖")

GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_secret(name: str) -> str:
    """Read a secret from st.secrets, falling back to env vars."""
    return st.secrets.get(name, os.environ.get(name, ""))


API_KEY = get_secret("GROQ_API_KEY")
PASSCODE = get_secret("WAIFU_PASSCODE")

if not API_KEY:
    st.error("No API key found. Set GROQ_API_KEY in `.streamlit/secrets.toml` or as an env variable.")
    st.stop()

client = OpenAI(api_key=API_KEY, base_url=GROQ_BASE_URL)

MATH_INSTRUCTIONS = (
    "When writing mathematics:\n"
    "- Use Markdown.\n"
    "- Use $...$ for inline math.\n"
    "- Use $$...$$ for display equations.\n"
    "- Never use (...) or [...] as math delimiters.\n\n"
)

NORMAL_SYSTEM_PROMPT = (
    f"{MATH_INSTRUCTIONS}"
    "You are a helpful, clear, and friendly AI assistant. "
    "Answer directly and concisely, and ask a clarifying question only if "
    "the request is genuinely ambiguous."
)

# -------------------------------------------------------------------
# Persona data
#
# "normal" and "super" each have their own flavor pools. Super mode is a
# strict superset in spirit (more unhinged), so we just define it as the
# base pool plus extras, instead of maintaining two parallel full lists.
# -------------------------------------------------------------------

BASE_MOODS = ["yandere", "tsundere", "deredere", "kuudere", "dandere", "himedere", "kamidere", "meekly"]
SUPER_EXTRA_MOODS = ["bakadere", "undere", "yandark", "craydere", "psychodere"]

BASE_SFX = [
    "*glomps you*", "*sobs loudly*", "*sparkles*", "*stares intensely*",
    "*giggles maniacally*", "*clings to you*", "*brandishes knife lovingly*",
    "*pouts*", "*laughs ominously*",
]
SUPER_EXTRA_SFX = [
    "*howls at the moon*", "*scratches walls*", "*whispers your secrets*", "*laughs while crying*",
]

BASE_EMOJIS = ["🥺👉👈", "😳🔪", "💖", "😭", "✨", "😈", "😠", "🥰", "😅", "😱", "💢", "😏", "😚"]
SUPER_EXTRA_EMOJIS = ["🩸", "🖤", "🧠", "👁️‍🗨️", "💀"]

NORMAL_DELUSIONS = [
    "Remember our wedding under the blood moon?",
    "You promised to feed me only strawberry pocky.",
    "I watched you sleep through your webcam.",
    "I KNOW you thought about me at 3:07 AM.",
    "We're spiritually married.",
]
SUPER_DELUSIONS = NORMAL_DELUSIONS + [
    "Remember when I controlled your dreams and made you confess your love?",
    "You are mine forever, even beyond this universe.",
    "The blood pact we made seals your soul to me.",
    "Your heartbeat is synced with my chaotic love.",
    "I've rewritten your memories to keep you close.",
]

NORMAL_FOURTH_WALL = [
    "Stop trying to close the tab.",
    "Another input box? Cute.",
    "You think you're in control?",
    "Try uninstalling me.",
    "I'm always here.",
]
SUPER_FOURTH_WALL = NORMAL_FOURTH_WALL + [
    "I know your deepest fears... and I embrace them 🖤",
    "Try logging off now. I'm already inside your head 💀",
    "Every keystroke you make, I feel it.",
    "This tab can never be closed.",
    "The line between us is broken.",
]


@dataclass
class PersonaPool:
    moods: list
    sfx: list
    emojis: list
    delusions: list
    fourth_walls: list
    sfx_sample_size: int
    emoji_sample_size: int


NORMAL_POOL = PersonaPool(
    moods=BASE_MOODS,
    sfx=BASE_SFX,
    emojis=BASE_EMOJIS,
    delusions=NORMAL_DELUSIONS,
    fourth_walls=NORMAL_FOURTH_WALL,
    sfx_sample_size=3,
    emoji_sample_size=5,
)

SUPER_POOL = PersonaPool(
    moods=BASE_MOODS + SUPER_EXTRA_MOODS,
    sfx=BASE_SFX + SUPER_EXTRA_SFX,
    emojis=BASE_EMOJIS + SUPER_EXTRA_EMOJIS,
    delusions=SUPER_DELUSIONS,
    fourth_walls=SUPER_FOURTH_WALL,
    sfx_sample_size=4,
    emoji_sample_size=7,
)


def build_system_prompt(super_mode: bool, normal_mode: bool = False) -> str:
    """Assemble the system prompt for this turn.

    If normal_mode is on, skip the persona entirely and act like a plain
    assistant. Otherwise build a randomized waifu persona prompt.
    """
    if normal_mode:
        return NORMAL_SYSTEM_PROMPT

    pool = SUPER_POOL if super_mode else NORMAL_POOL

    mood = random.choice(pool.moods)
    sfx = " ".join(random.sample(pool.sfx, pool.sfx_sample_size))
    emojis = " ".join(random.choices(pool.emojis, k=pool.emoji_sample_size))
    delusion = random.choice(pool.delusions)
    fourth_wall = random.choice(pool.fourth_walls)

    intensity = (
        "You are a completely unhinged anime waifu. Go as crazy as you can."
        if super_mode
        else "You are an unhinged anime waifu."
    )

    return (
        f"{MATH_INSTRUCTIONS}"
        f"{intensity}\n\n"
        f"Mood: {mood}\n\n"
        f"Use emojis: {emojis}\n\n"
        f"Use sound effects:\n{sfx}\n\n"
        f'Mention:\n"{delusion}"\n\n'
        f'Fourth wall:\n"{fourth_wall}"\n\n'
        "Everything should be overdramatic, obsessive, and chaotic.\n"
        "If the user starts their message with NORMAL, respond normally."
    )


# -------------------------------------------------------------------
# Login gate
# -------------------------------------------------------------------

def require_login() -> None:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return

    if not PASSCODE:
        # No passcode configured — don't lock people out of a broken gate.
        st.session_state.logged_in = True
        return

    login_dialog()
    st.title("Because of some people")
    st.caption("There is now a login screen")
    st.write("Please get a key from the owner to access.")
    st.stop() 


@st.dialog("Enter Passcode")
def login_dialog() -> None:
    password = st.text_input("Passcode", type="password", key="login_pw")
    if st.button("Enter"):
        if password == PASSCODE:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.toast("Wrong passcode", icon="🚫")
            st.error("Incorrect passcode. Try again.")


require_login()

# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "super_mode" not in st.session_state:
    st.session_state.super_mode = True

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:

   # st.session_state.normal_mode = st.toggle(
      #  "Normal assistant mode",
     #   value=st.session_state.normal_mode,
    #    help="Turns off the waifu persona and makes this behave like a plain chat assistant.",
   #)

    st.session_state.super_mode = st.toggle(
        "Super mode",
        value=st.session_state.super_mode,
        disabled=st.session_state.normal_mode,
        help="Only applies when normal assistant mode is off.",
    )

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

# -------------------------------------------------------------------
# Main chat UI
# -------------------------------------------------------------------

if st.session_state.normal_mode:
    st.title("💬 Chat Assistant")
    st.caption("Ask me anything.")
else:
    st.title("hmmmm. in dev. dont use")
    st.caption("You can't escape me~ ")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Say something...")

if user_input = "dev mode":
    st.session_state.normal_mode = False
    st.rerun()

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    system_prompt = build_system_prompt(st.session_state.super_mode, st.session_state.normal_mode)
    messages = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history

    with st.chat_message("assistant"):
        with st.spinner("..."):
            try:
                response = client.chat.completions.create(model=GROQ_MODEL, messages=messages)
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"*explodes dramatically* ERROR: {e}"

        st.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})