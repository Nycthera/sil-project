"""
Unhinged Waifu — a silly persona chatbot built on Streamlit + Groq.

Cleaned up from the original: dead code removed, persona data centralized,
config validated up front, and the request/response flow simplified.

Added: double-layer profanity filtering (ML score + rule-based) on user input.
Added: randomized comedic styles for normal assistant mode.
Added: /roast, /compliment, /8ball easter eggs.
"""

import os
import random
from dataclasses import dataclass, field

import streamlit as st
from openai import OpenAI
from profanity_check import predict_prob
from better_profanity import ProfanityFilter

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

# Rule-based filter, used as a second opinion alongside the ML score.
# Cached as a resource so it isn't rebuilt (it loads a wordlist/spacy
# model under the hood) on every Streamlit rerun.
@st.cache_resource
def get_rule_based_filter() -> ProfanityFilter:
    return ProfanityFilter()


pf = get_rule_based_filter()


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
    "a nature documentary narrator observing the user's habits with hushed fascination",
    "a conspiracy theorist who's convinced every bug is intentional sabotage",
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

COMPLIMENT_SYSTEM_PROMPT = (
    f"{MATH_INSTRUCTIONS}"
    "The user has explicitly typed /compliment, asking for the opposite of a roast. "
    "This is consensual and expected — commit fully.\n\n"
    "For this ONE reply only:\n"
    "- Open with a short, wildly over-the-top, sincere-sounding compliment about the "
    "user, based on whatever context is available in the conversation so far. If "
    "there's no context yet, compliment them for the sheer bravery of showing up.\n"
    "- Play it completely straight, like a nature documentary narrator or an awards "
    "show host — the humor comes from the excess, not from winking at the camera.\n"
    "- After the compliment, drop the act and go back to being a normal helpful "
    "assistant for any actual question in the message.\n"
    "- Keep the whole thing under ~120 words."
)

MAGIC_8BALL_ANSWERS = [
    "It is certain.", "Without a doubt.", "Yes, definitely.", "You may rely on it.",
    "As I see it, yes.", "Most likely.", "Outlook good.", "Signs point to yes.",
    "Reply hazy, try again.", "Ask again later.", "Better not tell you now.",
    "Cannot predict now.", "Concentrate and ask again.",
    "Don't count on it.", "My reply is no.", "My sources say no.",
    "Outlook not so good.", "Very doubtful.",
]

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


def build_system_prompt(
    super_mode: bool,
    normal_mode: bool = False,
    special: str | None = None,
) -> str:
    """Assemble the system prompt for this turn.

    `special` is a one-off override ("roast" or "compliment") that takes
    priority over everything else but doesn't change any persistent mode.
    Otherwise: normal_mode -> randomized funny-but-competent assistant,
    else -> randomized waifu persona.
    """
    if special == "roast":
        return ROAST_SYSTEM_PROMPT
    if special == "compliment":
        return COMPLIMENT_SYSTEM_PROMPT

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
# Profanity filter (double layer: ML score + rule-based)
# -------------------------------------------------------------------

def profanity_score(text: str) -> float:
    """Return a 0-1 probability that `text` contains profanity (ML model)."""
    if not text:
        return 0.0
    return float(predict_prob([text])[0])


def rule_based_is_profane(text: str) -> bool:
    """Second opinion via a rule/wordlist-based check, independent of the
    ML model above. Catches things the classifier wasn't trained on;
    the classifier in turn catches sneakier phrasing this misses.
    """
    if not text:
        return False
    return bool(pf.is_profane(text))


def check_profanity(text: str) -> tuple[bool, float, bool]:
    """Run both filters and decide whether to block.

    Returns (blocked, ml_score, rule_flagged) so the caller can show
    exactly why a message was blocked.
    """
    ml_score = profanity_score(text)
    rule_flagged = rule_based_is_profane(text)
    blocked = (ml_score >= PROFANITY_THRESHOLD) or rule_flagged
    return blocked, ml_score, rule_flagged


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
# UI helpers
# -------------------------------------------------------------------

def render_sidebar() -> None:
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


def render_title() -> None:
    if st.session_state.normal_mode:
        st.title("💬 Chat Assistant")
        st.caption("Ask me anything. (psst — try /roast, /compliment, /8ball)")
    else:
        st.title("hmmmm. in dev. dont use")
        st.caption("hmmm ")


def render_chat_history() -> None:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def handle_hidden_commands(user_input: str) -> bool:
    """Handle mode-switching commands that never touch the model.

    Returns True if the input was consumed as a command (caller should
    stop processing this input further).
    """
    if user_input == "dev mode":
        st.session_state.normal_mode = False
        st.rerun()
        return True

    if st.session_state.normal_mode is False and user_input == "norm mode":
        st.session_state.normal_mode = True
        st.rerun()
        return True

    return False


def handle_8ball(user_input: str) -> bool:
    """Instant, no-API-call easter egg. Returns True if handled."""
    if user_input.strip().lower().startswith("/8ball"):
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(f"🎱 {random.choice(MAGIC_8BALL_ANSWERS)}")
        # Not added to chat_history — it's a toy, not part of the real
        # conversation the model needs context on.
        return True
    return False


def call_model(messages: list) -> str:
    """Call the Groq-backed chat completion endpoint, with a persona-
    flavored fallback message if the request fails."""
    try:
        response = client.chat.completions.create(model=GROQ_MODEL, messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        return f"*explodes dramatically* ERROR: {e}"


def handle_user_message(user_input: str) -> None:
    blocked, ml_score, rule_flagged = check_profanity(user_input)

    if blocked:
        # Don't add it to chat_history and don't call the model — just
        # show it in this run so the user sees their own message and why
        # it was blocked.
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            reason = []
            if ml_score >= PROFANITY_THRESHOLD:
                reason.append(f"ML score {ml_score:.2f}")
            if rule_flagged:
                reason.append("rule-based filter")
            st.warning(f"Message blocked by profanity filter ({', '.join(reason)}).")
        return

    stripped = user_input.strip().lower()
    special = "roast" if stripped.startswith("/roast") else (
        "compliment" if stripped.startswith("/compliment") else None
    )

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    system_prompt = build_system_prompt(
        st.session_state.super_mode,
        st.session_state.normal_mode,
        special=special,
    )
    messages = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history

    with st.chat_message("assistant"):
        with st.spinner("..."):
            reply = call_model(messages)
        st.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

render_sidebar()
render_title()
render_chat_history()

user_input = st.chat_input("Say something...")

if user_input:
    if not handle_hidden_commands(user_input):
        if not handle_8ball(user_input):
            handle_user_message(user_input)
