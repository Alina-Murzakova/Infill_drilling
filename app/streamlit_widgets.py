from __future__ import annotations

from typing import Any

from app.streamlit_runtime import SESSION_DEFAULTS


def _format_default(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}".replace(".", ",")
    return str(value)


def default_value_help(key: str, extra: str | None = None) -> str:
    """Build a compact tooltip for a parameter initialized from session defaults."""
    default = SESSION_DEFAULTS.get(key)
    text = (
        f"Значение по умолчанию: {_format_default(default)}. "
        "Оно задаётся один раз при старте; затем поле можно изменить или очистить."
    )
    return f"{text}\n\n{extra}" if extra else text


def clearable_number_input(
    container,
    label: str,
    *,
    key: str,
    min_value: int | float,
    max_value: int | float,
    step: int | float,
    format: str | None = None,
    disabled: bool = False,
    required: bool = True,
    on_change=None,
    args: tuple[Any, ...] | None = None,
):
    """Render a number input that preserves a deliberately cleared value as ``None``."""
    default = SESSION_DEFAULTS.get(key)
    placeholder = (
        f"По умолчанию: {_format_default(default)}"
        if default is not None
        else ("Обязательное поле" if required and not disabled else "Не задано")
    )
    value = container.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=None,
        step=step,
        format=format,
        key=key,
        help=default_value_help(key),
        placeholder=placeholder,
        disabled=disabled,
        on_change=on_change,
        args=args,
        label_visibility="collapsed",
        persist_state="session",
    )
    if required and not disabled and value is None:
        container.caption(":red[:material/error: Обязательное поле]")
    return value
