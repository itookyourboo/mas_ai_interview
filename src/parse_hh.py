from dataclasses import dataclass
from functools import lru_cache

import requests
from bs4 import BeautifulSoup


@dataclass
class VacancyInfo:
    title: str
    experience: str
    company: str
    description: str
    skills: str


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15',
    'Accept': 'text/html',
}


def _validate_response(response):
    if not response.ok:
        raise requests.exceptions.RequestException(response.text)


def parse_html(html: str) -> VacancyInfo:
    soup = BeautifulSoup(html, 'html.parser')

    title = soup.find('h1', {'data-qa': 'vacancy-title'}).text
    experience = soup.find('span', {'data-qa': 'vacancy-experience'}).text
    company = soup.find('a', {'data-qa': 'vacancy-company-name'}).text
    description = soup.find('div', {'data-qa': 'vacancy-description'}).text
    skills = ', '.join(
        map(lambda item: item.text, soup.find_all('div', {'data-qa': 'bloko-tag bloko-tag_inline skills-element'}))
    )

    return VacancyInfo(
        title=title,
        experience=experience,
        company=company,
        description=description,
        skills=skills,
    )


@lru_cache(maxsize=256)
def parse_vacancy(url: str) -> VacancyInfo:
    response = requests.get(url, headers=HEADERS)
    _validate_response(response)
    return parse_html(response.text)
