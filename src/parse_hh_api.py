"""
Парсер вакансий hh.ru через публичный JSON API с OAuth2-авторизацией.

Документация:
- https://github.com/hhru/api
- https://api.hh.ru/openapi/redoc

Endpoint:     GET https://api.hh.ru/vacancies/{vacancy_id}
Token URL:    POST https://api.hh.ru/token (grant_type=client_credentials)

В отличие от scraping'а HTML (см. parse_hh.py), API:
- стабилен (поля не меняются при ребрендингах);
- возвращает enum-id для опыта/занятости (between1And3, full, ...);
- требует OAuth2-токен (Authorization: Bearer ...), глобально по спеке.

Получить креды можно тремя путями:
1. HH_ACCESS_TOKEN — готовый токен, скопированный из dev.hh.ru/admin (после
   первого выпуска токен показывается в админке приложения).
2. HH_CLIENT_ID + HH_CLIENT_SECRET — пара из dev.hh.ru, парсер сам выпустит
   токен через client_credentials и закэширует его в памяти процесса.
3. Без кредов парсер кидает VacancyAuthError с инструкцией.

Возвращает тот же VacancyInfo, что и parse_hh.py — drop-in альтернатива
для InterviewParams.from_vacancy.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache

import requests
from bs4 import BeautifulSoup

try:
    from parse_hh import VacancyInfo
except ModuleNotFoundError:  # pragma: no cover - fallback для запуска как src.*
    from src.parse_hh import VacancyInfo

try:
    import settings
except ModuleNotFoundError:  # pragma: no cover - fallback для запуска как src.*
    from src import settings


API_BASE_URL = "https://api.hh.ru"
TOKEN_URL = f"{API_BASE_URL}/token"
DEFAULT_TIMEOUT = 10  # секунд

# Маппинг enum-id опыта в текст с подстроками, которые ищет
# InterviewParams.from_vacancy ('без опыта', '1–3', '3–6').
EXPERIENCE_ID_TO_TEXT = {
    "noExperience": "Без опыта",
    "between1And3": "От 1 года до 3 лет (1–3)",
    "between3And6": "От 3 до 6 лет (3–6)",
    "moreThan6": "Более 6 лет",
}

VACANCY_ID_RE = re.compile(r"vacanc(?:y|ies)/(\d+)")

REGISTRATION_HINT = (
    "Зарегистрируйте приложение на https://dev.hh.ru/admin и пропишите "
    "HH_ACCESS_TOKEN (или HH_CLIENT_ID/HH_CLIENT_SECRET) в .env."
)


class VacancyAPIError(RuntimeError):
    """Ошибка обращения к hh.ru API."""


class VacancyAuthError(VacancyAPIError):
    """Не удалось авторизоваться (нет кредов, протух токен и т.п.)."""


# Кэш application access_token. Срок жизни — неограниченный по спеке hh.ru,
# но при 401/403 принудительно сбрасываем и перевыпускаем.
_token_lock = threading.Lock()
_cached_app_token: str | None = None


def _ua_headers() -> dict[str, str]:
    return {
        "User-Agent": settings.HH_USER_AGENT,
        "Accept": "application/json",
    }


def _fetch_app_token() -> str:
    """Запросить application access_token по client_credentials."""
    client_id = (settings.HH_CLIENT_ID or "").strip()
    client_secret = (settings.HH_CLIENT_SECRET or "").strip()
    if not client_id or not client_secret:
        raise VacancyAuthError(
            "Не заданы HH_CLIENT_ID и HH_CLIENT_SECRET. " + REGISTRATION_HINT
        )

    try:
        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            headers={**_ua_headers(), "Content-Type": "application/x-www-form-urlencoded"},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.exceptions.RequestException as exc:
        raise VacancyAPIError(f"Сеть недоступна при запросе токена: {exc}") from exc

    if not response.ok:
        raise VacancyAuthError(
            f"hh.ru token endpoint вернул {response.status_code}: {response.text[:200]}. "
            "Проверьте корректность HH_CLIENT_ID/HH_CLIENT_SECRET. "
            "Помните: токен можно запрашивать не чаще одного раза в 5 минут."
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise VacancyAPIError(f"Невалидный JSON от token endpoint: {exc}") from exc

    token = payload.get("access_token")
    if not token:
        raise VacancyAuthError(f"В ответе token endpoint нет access_token: {payload}")
    return token


def _get_access_token(force_refresh: bool = False) -> str:
    """
    Вернуть access_token приоритетно из env (HH_ACCESS_TOKEN), иначе выпустить
    через client_credentials и закэшировать.
    """
    env_token = (settings.HH_ACCESS_TOKEN or "").strip()
    if env_token:
        return env_token

    global _cached_app_token
    with _token_lock:
        if _cached_app_token and not force_refresh:
            return _cached_app_token
        _cached_app_token = _fetch_app_token()
        return _cached_app_token


def _invalidate_cached_token() -> None:
    global _cached_app_token
    with _token_lock:
        _cached_app_token = None


def extract_vacancy_id(url_or_id: str) -> str:
    """
    Достать числовой id вакансии из URL hh.ru или вернуть строку как есть,
    если она уже выглядит как id.

    Поддерживаются формы:
        - https://hh.ru/vacancy/12345
        - https://hh.ru/vacancy/12345?from=...
        - https://spb.hh.ru/vacancy/12345
        - https://api.hh.ru/vacancies/12345
        - 12345
    """
    value = (url_or_id or "").strip()
    if value.isdigit():
        return value
    match = VACANCY_ID_RE.search(value)
    if not match:
        raise VacancyAPIError(f"Не удалось извлечь id вакансии из {url_or_id!r}")
    return match.group(1)


def _html_to_text(html: str) -> str:
    """Преобразовать HTML-описание в plain text с пробелами между блоками."""
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def _experience_to_text(experience: dict | None) -> str:
    if not experience:
        return ""
    exp_id = experience.get("id") or ""
    if exp_id in EXPERIENCE_ID_TO_TEXT:
        return EXPERIENCE_ID_TO_TEXT[exp_id]
    return experience.get("name") or ""


def _skills_to_text(key_skills: list[dict] | None) -> str:
    if not key_skills:
        return ""
    names = [item.get("name", "").strip() for item in key_skills if item]
    return ", ".join(name for name in names if name)


def _is_token_error(response: requests.Response) -> bool:
    """Похож ли HTTP-ответ на проблему с токеном (требует перевыпуск)."""
    if response.status_code not in (401, 403):
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    errors = payload.get("errors", []) or []
    token_error_types = {"bad_authorization", "token_expired", "token_revoked"}
    return any((err or {}).get("type") in token_error_types for err in errors)


def _get_with_auth(url: str, *, allow_token_refresh: bool = True) -> requests.Response:
    """GET с Bearer-токеном; на token-ошибке пробует один раз перевыпустить токен."""
    token = _get_access_token()
    try:
        response = requests.get(
            url,
            headers={**_ua_headers(), "Authorization": f"Bearer {token}"},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.exceptions.RequestException as exc:
        raise VacancyAPIError(f"Ошибка сети при запросе {url}: {exc}") from exc

    if allow_token_refresh and _is_token_error(response):
        _invalidate_cached_token()
        return _get_with_auth(url, allow_token_refresh=False)
    return response


def _fetch_vacancy_json(vacancy_id: str) -> dict:
    url = f"{API_BASE_URL}/vacancies/{vacancy_id}"
    response = _get_with_auth(url)

    if response.status_code == 404:
        raise VacancyAPIError(f"Вакансия {vacancy_id} не найдена или скрыта")
    if response.status_code in (401, 403):
        raise VacancyAuthError(
            f"hh.ru API вернул {response.status_code}: {response.text[:200]}. "
            + REGISTRATION_HINT
        )
    if response.status_code == 429:
        raise VacancyAPIError("hh.ru API: слишком много запросов (429), попробуйте позже")
    if not response.ok:
        raise VacancyAPIError(
            f"hh.ru API вернул {response.status_code}: {response.text[:200]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise VacancyAPIError(f"Невалидный JSON от hh.ru API: {exc}") from exc


def parse_vacancy_payload(payload: dict) -> VacancyInfo:
    """Преобразовать JSON-ответ API в VacancyInfo (для тестов и оффлайн-парсинга)."""
    title = payload.get("name", "") or ""
    experience = _experience_to_text(payload.get("experience"))
    company = (payload.get("employer") or {}).get("name", "") or ""
    description = _html_to_text(payload.get("description", "") or "")
    skills = _skills_to_text(payload.get("key_skills"))

    return VacancyInfo(
        title=title,
        experience=experience,
        company=company,
        description=description,
        skills=skills,
    )


@lru_cache(maxsize=256)
def parse_vacancy(url_or_id: str) -> VacancyInfo:
    """
    Загрузить и распарсить вакансию hh.ru через публичный API.

    На вход — либо полный URL вакансии (https://hh.ru/vacancy/...),
    либо просто числовой id.
    """
    vacancy_id = extract_vacancy_id(url_or_id)
    payload = _fetch_vacancy_json(vacancy_id)
    return parse_vacancy_payload(payload)
