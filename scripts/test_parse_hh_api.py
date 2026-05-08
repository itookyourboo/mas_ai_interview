"""
Smoke-тест для парсера вакансий через hh.ru API (с OAuth2-авторизацией).

Принимает URL вакансии (или просто id) и печатает поля VacancyInfo
плюс производное mapping в InterviewParams.

Перед запуском нужны креды:
    HH_ACCESS_TOKEN=...               # готовый токен из dev.hh.ru/admin
ИЛИ
    HH_CLIENT_ID=...
    HH_CLIENT_SECRET=...

Запуск:
    uv run python scripts/test_parse_hh_api.py https://hh.ru/vacancy/12345678
    uv run python scripts/test_parse_hh_api.py 12345678
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import settings
from src.parse_hh_api import (
    VacancyAPIError,
    VacancyAuthError,
    extract_vacancy_id,
    parse_vacancy,
)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _auth_status_line() -> str:
    if settings.HH_ACCESS_TOKEN:
        masked = settings.HH_ACCESS_TOKEN[:6] + "…" + settings.HH_ACCESS_TOKEN[-4:]
        return f"HH_ACCESS_TOKEN set ({masked})"
    if settings.HH_CLIENT_ID and settings.HH_CLIENT_SECRET:
        return f"HH_CLIENT_ID/SECRET set (client_id={settings.HH_CLIENT_ID[:6]}…)"
    return "[!] креды не заданы — запрос упадёт с VacancyAuthError"


def main() -> int:
    parser = argparse.ArgumentParser(description="Test hh.ru API vacancy parser")
    parser.add_argument("url", help="URL вакансии hh.ru или просто id")
    parser.add_argument(
        "--full-description",
        action="store_true",
        help="Печатать описание целиком (по умолчанию обрезается до 500 символов)",
    )
    args = parser.parse_args()

    print(f"Auth: {_auth_status_line()}")

    try:
        vacancy_id = extract_vacancy_id(args.url)
    except VacancyAPIError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"Vacancy id: {vacancy_id}")
    print(f"API url:    https://api.hh.ru/vacancies/{vacancy_id}")
    print("-" * 60)

    try:
        vacancy = parse_vacancy(args.url)
    except VacancyAuthError as exc:
        print(f"[AUTH ERROR] {exc}")
        return 2
    except VacancyAPIError as exc:
        print(f"[ERROR] {exc}")
        return 1

    description = vacancy.description if args.full_description else _truncate(vacancy.description, 500)

    print(f"title:       {vacancy.title}")
    print(f"company:     {vacancy.company}")
    print(f"experience:  {vacancy.experience}")
    print(f"skills:      {vacancy.skills or '(не указаны)'}")
    print(f"description: {description}")

    try:
        from src.main import InterviewParams
    except Exception as exc:
        print(f"\n[warning] не удалось построить InterviewParams: {exc}")
        return 0

    params = InterviewParams.from_vacancy(vacancy)
    print("\n" + "=" * 60)
    print("Производный InterviewParams:")
    print(f"  position:   {params.position}")
    print(f"  level:      {params.level}")
    print(f"  tech_stack: {params.tech_stack}")
    print(f"  topics:     {params.topics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
