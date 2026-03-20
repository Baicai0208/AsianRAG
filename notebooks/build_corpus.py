import requests
from bs4 import BeautifulSoup
import json
import time
import urllib.parse

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def get_soup(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        return BeautifulSoup(res.text, 'html.parser')
    except:
        return None

def clean_text(soup):
    """
    修复：原版用 ' '.join(...) 把所有段落压成一行，
    导致 sentence/paragraph chunking 策略无法工作。
    现在改为用 '\n\n' 连接段落，保留段落边界。
    同时过滤掉太短的段落（噪声），避免引入无意义碎片。
    """
    if not soup:
        return ""
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'table']):
        tag.decompose()

    paragraphs = []
    for p in soup.find_all('p'):
        text = p.get_text().strip()
        # 过滤掉太短的段落（少于 30 字符，通常是按钮/标签等噪声）
        if len(text) >= 30:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)

def get_wiki_links():
    url = "https://en.wikipedia.org/wiki/List_of_Asian_cuisines"
    soup = get_soup(url)
    links = []
    keywords = ['chinese', 'japanese', 'korean', 'taiwanese', 'hong_kong', 'macanese']
    if soup:
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            if href.startswith('/wiki/') and 'cuisine' in href and any(k in href for k in keywords):
                links.append(urllib.parse.urljoin(url, a['href']))
    return links

def get_wikibooks_links():
    url = "https://en.wikibooks.org/wiki/Cookbook:Cuisines"
    soup = get_soup(url)
    links = []
    if soup:
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.text.lower()
            if href.startswith('/wiki/Cookbook:') and any(k in text for k in ['china', 'chinese', 'japan', 'japanese', 'korea', 'korean', 'taiwan']):
                links.append(urllib.parse.urljoin(url, href))
    return links

def get_blog_links():
    url = "https://aroundtheworldin80cuisinesblog.wordpress.com/"
    soup = get_soup(url)
    links = []
    if soup:
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            if 'aroundtheworldin80cuisinesblog' in href and any(k in href for k in ['china', 'japan', 'korea']):
                links.append(a['href'])
    return links

def main():
    all_links = list(set(get_wiki_links() + get_wikibooks_links() + get_blog_links()))
    corpus = []
    
    for link in all_links:
        print(f"Scraping: {link}")
        soup = get_soup(link)
        text = clean_text(soup)
        if text:
            corpus.append({"source": link, "text": text})
        time.sleep(1)
        
    with open('../data/corpus/east_asian_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Successfully scraped {len(corpus)} pages.")

if __name__ == "__main__":
    main()