"""Load the project's API key without overriding an existing shell setting."""
import os
from pathlib import Path


def load_project_env(path: Path | None = None) -> None:
    path = path if path is not None else Path(__file__).resolve().parent.parent / '.env'
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, separator, value = line.partition('=')
        if not separator or name.strip() != 'GEMINI_API_KEY':
            continue
        value = value.strip()
        if value.startswith(('"', "'")):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0 or (value[end+1:].strip() and not value[end+1:].strip().startswith('#')):
                raise ValueError('Invalid quoted GEMINI_API_KEY in .env')
            value = value[1:end]
        else:
            value = value.split('#', 1)[0].strip()
        if value:
            os.environ.setdefault('GEMINI_API_KEY', value)
