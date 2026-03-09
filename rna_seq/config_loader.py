from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yml")


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML configuration file for RNA-seq scripts.

    Args:
        config_path (Path | str): Path to the YAML configuration file.

    Returns:
        dict[str, Any]: Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping at the top level.")
    return loaded


def get_config_section(
    config: dict[str, Any],
    section: str,
    required_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Return one configuration section and validate required keys.

    Args:
        config (dict[str, Any]): Full parsed configuration dictionary.
        section (str): Top-level section name to retrieve.
        required_keys (list[str] | None): Optional keys that must exist in the section.

    Returns:
        dict[str, Any]: Requested configuration section.
    """
    if section not in config:
        raise KeyError(f"Missing required config section: '{section}'")

    section_value = config[section]
    if not isinstance(section_value, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")

    if required_keys:
        missing = [key for key in required_keys if key not in section_value]
        if missing:
            missing_str = ", ".join(missing)
            raise KeyError(
                f"Config section '{section}' is missing required key(s): {missing_str}"
            )
    return section_value


def render_template(template: str, values: dict[str, Any]) -> str:
    """Format a template string with config-provided values.

    Args:
        template (str): Template string with ``str.format`` placeholders.
        values (dict[str, Any]): Values used to fill placeholders.

    Returns:
        str: Formatted string.
    """
    try:
        return template.format(**values)
    except KeyError as exc:
        missing_key = exc.args[0]
        raise KeyError(
            f"Missing template value '{missing_key}' while formatting '{template}'."
        ) from exc
