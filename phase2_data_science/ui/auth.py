"""User authentication for the dashboard: register, sign in, sign out, guest access.

Passwords are never stored. Each account keeps a random salt and a PBKDF2-HMAC-SHA256 hash
(200,000 rounds) in users.json, which is listed in .gitignore.

Turn the login screen off with the environment variable PHASE2_AUTH=off.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path

USERS_FILE = Path(__file__).resolve().parents[1] / "users.json"
ROUNDS = 200_000
MAX_ATTEMPTS = 5


def enabled() -> bool:
    return os.environ.get("PHASE2_AUTH", "on").lower() not in ("off", "0", "false", "no")


def _load() -> dict:
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(users: dict) -> None:
    tmp = USERS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(users, indent=2), encoding="utf-8")
    tmp.replace(USERS_FILE)                                # atomic: never leaves a half-written file


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ROUNDS).hex()


def password_problem(password: str) -> str | None:
    if len(password) < 8:
        return "Use at least 8 characters."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return "Use both letters and numbers."
    return None


def register(username: str, password: str, full_name: str = "") -> tuple[bool, str]:
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,30}", username):
        return False, "Username: 3-30 characters, letters, numbers, dot, dash or underscore."
    problem = password_problem(password)
    if problem:
        return False, problem
    users = _load()
    if username in users:
        return False, "That username is taken."
    salt = secrets.token_bytes(16)
    users[username] = {"name": full_name.strip() or username, "salt": salt.hex(), "hash": _hash(password, salt),
                       "role": "admin" if not users else "analyst",          # the first account administers the app
                       "created": datetime.now().isoformat(timespec="seconds")}
    _save(users)
    return True, "Account created."


def verify(username: str, password: str) -> dict | None:
    user = _load().get(username.strip().lower())
    if not user:
        _hash(password, b"0" * 16)                         # same work either way: no hint whether the user exists
        return None
    ok = hmac.compare_digest(_hash(password, bytes.fromhex(user["salt"])), user["hash"])
    return {"username": username.strip().lower(), "name": user["name"], "role": user["role"]} if ok else None


def gate() -> None:
    """Show the sign-in screen until someone signs in or continues as a guest."""
    import streamlit as st

    if not enabled() or st.session_state.get("user"):
        return
    st.markdown("<div style='max-width:460px;margin:6vh auto 0'>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.title("Data Science · Phase 2")
        st.caption("Sign in to open the dashboard.")
        tab_in, tab_new = st.tabs(["Sign in", "Create account"])
        with tab_in:
            with st.form("login"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                go = st.form_submit_button("Sign in", type="primary", width="stretch")
            if go:
                tries = st.session_state.get("login_tries", 0)
                if tries >= MAX_ATTEMPTS:
                    st.error("Too many failed attempts. Reload the page to try again.")
                else:
                    user = verify(u, p)
                    if user:
                        st.session_state["user"] = user
                        st.session_state["login_tries"] = 0
                        st.rerun()
                    st.session_state["login_tries"] = tries + 1
                    st.error(f"Wrong username or password ({MAX_ATTEMPTS - tries - 1} attempts left).")
        with tab_new:
            with st.form("register"):
                name = st.text_input("Full name")
                u2 = st.text_input("Choose a username")
                p2 = st.text_input("Choose a password", type="password", help="At least 8 characters, with letters and numbers")
                p3 = st.text_input("Repeat the password", type="password")
                make = st.form_submit_button("Create account", width="stretch")
            if make:
                if p2 != p3:
                    st.error("The two passwords do not match.")
                else:
                    ok, msg = register(u2, p2, name)
                    (st.success if ok else st.error)(msg + (" You can sign in now." if ok else ""))
        st.divider()
        if st.button("Continue as guest", width="stretch"):
            st.session_state["user"] = {"username": "guest", "name": "Guest", "role": "guest"}
            st.rerun()
        st.caption("Passwords are stored only as salted PBKDF2 hashes.")
    st.stop()


def sidebar_badge() -> None:
    import streamlit as st

    user = st.session_state.get("user")
    if not enabled() or not user:
        return
    with st.sidebar:
        st.caption(f"Signed in as **{user['name']}** ({user['role']})")
        if st.button("Sign out", key="signout"):
            st.session_state.pop("user", None)
            st.rerun()
