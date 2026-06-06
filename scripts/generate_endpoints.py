import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import (
    RASA_ACTION_URL,
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
)

TEMPLATE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "rasa_bot", "endpoints.yml"
))


def generate():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace("__RASA_ACTION_URL__", RASA_ACTION_URL)
    content = content.replace("__DB_HOST__", DB_HOST)
    content = content.replace("__DB_PORT__", str(DB_PORT))
    content = content.replace("__DB_NAME__", DB_NAME)
    content = content.replace("__DB_USER__", DB_USER)
    content = content.replace("__DB_PASSWORD__", DB_PASSWORD)

    with open(TEMPLATE_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[generate_endpoints] endpoints.yml updated with env values.")


if __name__ == "__main__":
    generate()
