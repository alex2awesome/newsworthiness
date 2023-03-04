from bs4 import BeautifulSoup
import re
import os

def specific_prefetch_filtering(cdx_results):
    pass


def score_webpage(page, raw_html, file_on_disk, *args, **kwargs):
    """Determine whether we should include the webpage or not.

    Reasons for not including the web-page:
        * the layout is a mobile-only layout.
        * redirects or other blank pages
        * more TK
    """
    # handle redirect
    key_strings = [
        'You have sent too many requests in a given amount of time',
    ]
    if os.path.getsize(file_on_disk) < 100:
        return False

    if len(raw_html) < 100:
        return False

    soup = BeautifulSoup(raw_html, features='lxml')
    raw_text = soup.get_text()
    text = re.sub('\s+', ' ', raw_text)
    matches = any(map(lambda x: x in text, key_strings))
    return not matches
