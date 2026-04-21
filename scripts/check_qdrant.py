import json

from services.semantic_service import qdrant_healthcheck


def main() -> None:
    status = qdrant_healthcheck()
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
