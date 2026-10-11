def render_rows(rows: list[dict], template: str) -> list[dict]:
    if not isinstance(template, str) or "{text}" not in template:
        raise ValueError("prompt_template must be a string containing the {text} field")
    rendered = []
    for row in rows:
        try:
            text = template.format(text=row["text"])
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError(f"Invalid prompt_template {template!r}: {exc}") from exc
        rendered.append({**row, "text": text})
    return rendered
