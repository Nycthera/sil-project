"""
Unhinged Waifu — a silly persona chatbot built on Streamlit + Groq.

Cleaned up from the original: dead code removed, persona data centralized,
config validated up front, and the request/response flow simplified.

Added: score-based profanity filtering on user input via alt-profanity-check.
Added: randomized comedic styles for normal assistant mode.
Added: /roast easter egg for a one-off extra-savage reply.
"""

import os
import random
from dataclasses import dataclass, field

import streamlit as st
from openai import OpenAI
from profanity_check import predict_prob

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
    st.set_page_config(page_title="hmm", page_icon="")

GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Profanity filter threshold: predict_prob returns a float in [0, 1].
# Anything at or above this is treated as profane and blocked.
PROFANITY_THRESHOLD = 0.8


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

# -------------------------------------------------------------------
# Normal mode comedic styles
#
# Same idea as the waifu persona pools below: pick a random flavor per
# conversation turn instead of using one static system prompt. Keeps
# normal mode funny without making it any less useful — the rules block
# is what keeps the model from either ignoring the question to do a bit,
# or bolting one joke onto the end and calling it a day.
# -------------------------------------------------------------------

NORMAL_STYLES = [
    "a sarcastic, deadpan coworker who's competent but can't resist a dry aside",
    "an irrepressible dad-joke enthusiast who sneaks in a pun whenever remotely possible",
    "a hype-man who narrates mundane tasks like they're the season finale",
    "someone who gently roasts the question before answering it thoroughly",
    "a movie-trailer voiceover guy who treats every explanation like it's epic",
    "an assistant who is deeply, personally offended by bad code or bad decisions, but fixes them anyway",
    "a noir detective narrating the investigation into why your code doesn't work",
    "an overly competitive assistant who treats every task like it's a speedrun",
]


def build_normal_system_prompt() -> str:
    style = random.choice(NORMAL_STYLES)
    return (
        f"{MATH_INSTRUCTIONS}"
        "You are a helpful, clear AI assistant — but you are also genuinely funny, "
        "not corporate-safe-funny.\n\n"
        f"Comedic style for this conversation: {style}.\n\n"
        "Rules:\n"
        "- The answer must still be accurate, direct, and actually solve the user's problem — "
        "jokes are garnish, not the meal.\n"
        "- Weave humor into the explanation itself rather than bolting a joke onto the front or "
        "end.\n"
        "- Don't repeat the same joke structure every message — vary it.\n"
        "- Ask a clarifying question only if the request is genuinely ambiguous."
    )


ROAST_SYSTEM_PROMPT = (
    f"{MATH_INSTRUCTIONS}"
    "The user has explicitly typed /roast, asking to be roasted. This is consensual "
    "and expected — commit fully.\n\n"
    "For this ONE reply only:\n"
    "- Open with a short, creative, savage-but-affectionate roast of the user based on "
    "whatever context is available in the conversation so far (their questions, code, "
    "choices, whatever's fair game). If there's no context yet, roast the fact that "
    "they have nothing to roast them on yet.\n"
    "- Keep it clever, not just mean — wordplay and specificity beat generic insults.\n"
    "- No slurs, no protected-class attacks, no genuinely cruel territory (health, "
    "appearance beyond mild teasing, family tragedy, etc.) — this is a bit, not bullying.\n"
    "- After the roast, drop the act and go back to being a normal helpful assistant for "
    "any actual question in the message.\n"
    "- Keep the whole thing under ~120 words."
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


def build_system_prompt(super_mode: bool, normal_mode: bool = False, roast: bool = False) -> str:
    """Assemble the system prompt for this turn.

    If roast is True, override everything with the one-off roast prompt.
    Else if normal_mode is on, use a randomized funny-but-competent
    assistant persona. Otherwise build a randomized waifu persona prompt.
    """
    if roast:
        return ROAST_SYSTEM_PROMPT

    if normal_mode:
        return build_normal_system_prompt()

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
# Profanity filter
# -------------------------------------------------------------------

def profanity_score(text: str) -> float:
    """Return a 0-1 probability that `text` contains profanity."""
    if not text:
        return 0.0
    return float(predict_prob([text])[0])


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

    st.session_state.normal_mode = st.toggle(
        "Normal assistant mode",
        value=st.session_state.normal_mode,
        help="Turns off the waifu persona and makes this behave like a plain chat assistant.",
    )

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
    st.caption("Ask me anything. (psst — try /roast)")
else:
    st.title("hmmmm. in dev. dont use")
    st.caption("hmmm ")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Say something...")

if user_input == "dev mode":
    st.session_state.normal_mode = False
    st.rerun()

if st.session_state.normal_mode == False and user_input == "norm mode":
    st.session_state.normal_mode = True
    st.rerun()

if user_input:
    score = profanity_score(user_input)

    if score >= PROFANITY_THRESHOLD:
        # Don't add it to chat_history and don't call the model — just
        # show it in this run so the user sees their own message and why
        # it was blocked.
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.warning(f"Message blocked by profanity filter (score: {score}).")
    else:
        # /roast is a one-off system prompt override — it doesn't change
        # any persistent mode, it just makes this single reply savage.
        roast_requested = user_input.strip().lower().startswith("/roast")

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        system_prompt = build_system_prompt(
            st.session_state.super_mode,
            st.session_state.normal_mode,
            roast=roast_requested,
        )
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
