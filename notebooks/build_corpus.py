import requests
from bs4 import BeautifulSoup
import json
import time
import re
import urllib.parse
from urllib.parse import urljoin

# ── Ethical scraping header ──
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
        "(Student Project/Data Collection for RAG)"
    )
}

# ── 额外补充的 Wikipedia 页面（自动抓取可能遗漏） ──
EXTRA_WIKI_PAGES = [
    "https://en.wikipedia.org/wiki/History_of_Japanese_cuisine",
    "https://en.wikipedia.org/wiki/Korean_royal_court_cuisine",
    "https://en.wikipedia.org/wiki/Chinese_imperial_cuisine",
    "https://en.wikipedia.org/wiki/Taiwanese_cuisine",
    "https://en.wikipedia.org/wiki/Hong_Kong_cuisine",
    "https://en.wikipedia.org/wiki/Macanese_cuisine",
    "https://en.wikipedia.org/wiki/Japanese_cuisine",
    "https://en.wikipedia.org/wiki/Korean_cuisine",
    "https://en.wikipedia.org/wiki/Chinese_cuisine",
]

# ── Wikibooks 菜系入口页（第一层，硬编码保底） ──
WIKIBOOKS_CUISINE_PAGES = [
    "https://en.wikibooks.org/wiki/Cookbook:Chinese_cuisine",
    "https://en.wikibooks.org/wiki/Cookbook:Cuisine_of_Japan",
    "https://en.wikibooks.org/wiki/Cookbook:Cuisine_of_Korea",
    "https://en.wikibooks.org/wiki/Cookbook:Sushi",
    "https://en.wikibooks.org/wiki/Cookbook:Kimchi",
    "https://en.wikibooks.org/wiki/Cookbook:Mapo_Tofu",
    "https://en.wikibooks.org/wiki/Cookbook:Fried_Rice",
    "https://en.wikibooks.org/wiki/Cookbook:Miso_Soup",
    "https://en.wikibooks.org/wiki/Cookbook:Teriyaki_Chicken",
    "https://en.wikibooks.org/wiki/Cookbook:Daifuku",
    "https://en.wikibooks.org/wiki/Cookbook:Jangjorim",
]

# ── 博客分类页（直接指定，不依赖首页动态） ──
BLOG_CATEGORY_URLS = [
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/32-southern-china/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/65-northern-china/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/12-japan/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/71-korea/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/taiwan/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/hong-kong/",
]


# ══════════════════════════════════════════
#  内容提取函数
# ══════════════════════════════════════════

def get_soup(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        print(f"  ❌ 请求失败 {url}: {e}")
        return None


def clean_mediawiki(soup):
    """
    针对 Wikipedia / Wikibooks 的专项清洗（移植自 spider.py）。
    删除 navbox、infobox、references 等噪声节点后提取正文。
    返回用 \n\n 连接的段落字符串。
    """
    if not soup:
        return ""

    content_div = soup.find('div', class_='mw-parser-output')
    if not content_div:
        return ""

    # 销毁所有干扰组件
    junk_selectors = [
        'div.navbox',
        'div.toc',
        'table.infobox',
        'table.wikitable',
        'span.mw-editsection',
        'sup.reference',
        'style', 'script',
        'table.vertical-navbox',
        'div.portalbox',
        'ol.references',
        'div.reflist',
        'div.hatnote',
    ]
    for selector in junk_selectors:
        for node in content_div.select(selector):
            node.decompose()

    # 提取正文段落（p 为主，h2/h3 保留结构感）
    paragraphs = []
    for element in content_div.find_all(['p', 'h2', 'h3']):
        text = element.get_text(separator=' ', strip=True)
        if len(text) >= 30:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def clean_blog(soup):
    """
    针对 WordPress 博客的提取逻辑（移植自 spider.py entry-content 方式）。
    优先找 entry-content div，fallback 到所有 <p>。
    返回用 \n\n 连接的段落字符串。
    """
    if not soup:
        return ""

    # 尝试找 WordPress 标准正文容器
    content_div = soup.find('div', class_='entry-content')
    if not content_div:
        # Fallback：直接找 article 标签
        content_div = soup.find('article')
    if not content_div:
        content_div = soup  # 最终 fallback

    paragraphs = []
    for element in content_div.find_all(['p', 'li', 'h3']):
        text = element.get_text(separator=' ', strip=True)
        if len(text) >= 30:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


# ══════════════════════════════════════════
#  链接收集函数
# ══════════════════════════════════════════

def get_wiki_links():
    """
    从 List_of_Asian_cuisines 的 East_Asian_cuisine section
    精准提取东亚菜系词条链接（移植自 spider.py 的 anchor 定位逻辑）。
    """
    base_url = "https://en.wikipedia.org/wiki/List_of_Asian_cuisines"
    soup = get_soup(base_url)
    links = []
    if not soup:
        return links

    east_asian_span = soup.find(id="East_Asian_cuisine")
    if east_asian_span:
        ul_node = east_asian_span.parent.find_next_sibling('ul')
        if ul_node:
            for a in ul_node.find_all('a'):
                href = a.get('href', '')
                if href.startswith('/wiki/'):
                    links.append(urljoin(base_url, href))

    # 关键词 fallback：防止 anchor 结构变化时完全失效
    keywords = ['chinese', 'japanese', 'korean', 'taiwanese', 'hong_kong', 'macanese']
    for a in soup.find_all('a', href=True):
        href = a['href'].lower()
        if (href.startswith('/wiki/')
                and 'cuisine' in href
                and any(k in href for k in keywords)):
            full = urljoin(base_url, a['href'])
            if full not in links:
                links.append(full)

    return links


def get_wikibooks_links():
    """
    两层抓取：
      第一层：从 Cuisines 总目录 + 硬编码入口找到菜系子页。
      第二层：进入每个菜系子页，提取所有具体食谱链接。
    硬编码的已知重要食谱 URL 作为额外保底。
    """
    wikibooks_base = "https://en.wikibooks.org"
    cuisine_pages = set(WIKIBOOKS_CUISINE_PAGES)

    # 第一层：从总目录页发现更多菜系入口
    index_soup = get_soup("https://en.wikibooks.org/wiki/Cookbook:Cuisines")
    if index_soup:
        for a in index_soup.find_all('a', href=True):
            href = a['href']
            text = a.text.lower()
            if href.startswith('/wiki/Cookbook:') and any(
                k in text for k in ['china', 'chinese', 'japan', 'japanese',
                                     'korea', 'korean', 'taiwan']
            ):
                cuisine_pages.add(urljoin(wikibooks_base, href))
    time.sleep(1)

    # 第二层：进入每个菜系页提取具体食谱链接
    all_links = set(cuisine_pages)
    for cuisine_url in cuisine_pages:
        print(f"  [Wikibooks] Scanning: {cuisine_url}")
        soup = get_soup(cuisine_url)
        if not soup:
            time.sleep(1)
            continue
        for a in soup.find_all('a', href=True):
            href = a['href']
            if (href.startswith('/wiki/Cookbook:')
                    and '#' not in href
                    and 'File:' not in href
                    and 'Special:' not in href):
                all_links.add(urljoin(wikibooks_base, href))
        time.sleep(1)

    return list(all_links)


def get_blog_links():
    """
    从硬编码的分类页收集博客文章链接。
    同时用 spider.py 的关键词动态匹配作为补充，
    应对博客分类结构可能发生变化的情况。
    """
    east_asian_keywords = ["China", "Japan", "South Korea", "North Korea",
                            "Taiwan", "Mongolia", "Macau", "Hong Kong"]
    links = set()

    # 主策略：直接遍历分类页
    for cat_url in BLOG_CATEGORY_URLS:
        print(f"  [Blog] Scanning category: {cat_url}")
        soup = get_soup(cat_url)
        if not soup:
            time.sleep(1)
            continue
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'aroundtheworldin80cuisinesblog' in href.lower():
                links.add(href)
        time.sleep(1)

    # 备用策略：从首页侧边栏动态匹配关键词（移植自 spider.py）
    base_url = "https://aroundtheworldin80cuisinesblog.wordpress.com/"
    soup = get_soup(base_url)
    if soup:
        pattern = r'\b(?:' + '|'.join(east_asian_keywords) + r')\b'
        for a in soup.find_all('a', href=True):
            text = a.get_text(strip=True)
            href = a['href']
            if re.search(pattern, text, re.IGNORECASE):
                if 'aroundtheworldin80cuisinesblog' in href or href.startswith('/'):
                    links.add(urljoin(base_url, href))
    time.sleep(1)

    return list(links)


# ══════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════

def main():
    print("=" * 50)
    print("Step 1: Collecting links...")
    print("=" * 50)

    wiki_links = get_wiki_links()
    print(f"  Wikipedia (auto):     {len(wiki_links)} links")

    wikibooks_links = get_wikibooks_links()
    print(f"  Wikibooks (2-layer):  {len(wikibooks_links)} links")

    blog_links = get_blog_links()
    print(f"  Blog:                 {len(blog_links)} links")

    all_links = list(set(wiki_links + wikibooks_links + blog_links + EXTRA_WIKI_PAGES))
    print(f"\n  Total unique links: {len(all_links)}\n")

    print("=" * 50)
    print("Step 2: Scraping pages...")
    print("=" * 50)

    corpus = []
    for i, link in enumerate(all_links):
        print(f"[{i+1}/{len(all_links)}] {link}")
        soup = get_soup(link)
        if not soup:
            time.sleep(1)
            continue

        # 根据域名选择对应的提取策略
        if 'wikipedia.org' in link or 'wikibooks.org' in link:
            text = clean_mediawiki(soup)
        else:
            text = clean_blog(soup)

        if text:
            corpus.append({"source": link, "text": text})
            print(f"  ✅ {len(text)} chars")
        else:
            print(f"  ⚠️  No content extracted")

        time.sleep(1)

    print("\n" + "=" * 50)
    print("Step 3: Saving corpus...")
    print("=" * 50)

    output_path = '../data/corpus/east_asian_corpus.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Done. Scraped {len(corpus)} pages → {output_path}")


if __name__ == "__main__":
    main()