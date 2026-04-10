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

# ── 博客分类页（直接指定，不依赖首页动态） ──
BLOG_CATEGORY_URLS = [
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/32-southern-china/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/65-northern-china/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/12-japan/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/71-korea/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/taiwan/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/hong-kong/",
]

# ── Wikibooks 底层分类页（借鉴对方脚本：直接绕过前端折叠，访问 Category 页） ──
WIKIBOOKS_CATEGORY_URLS = [
    "https://en.wikibooks.org/wiki/Category:Chinese_recipes",
    "https://en.wikibooks.org/wiki/Category:Japanese_recipes",
    "https://en.wikibooks.org/wiki/Category:Korean_recipes",
    "https://en.wikibooks.org/wiki/Category:Taiwanese_recipes",
    "https://en.wikibooks.org/wiki/Category:Hong_Kong_recipes",
]

# ── 保底的 Wikibooks 入口页（原有逻辑保留，作为补充） ──
WIKIBOOKS_ENTRY_PAGES = [
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
    针对 Wikipedia / Wikibooks 的专项清洗。
    删除 navbox、infobox、references 等噪声节点后提取正文。
    返回用 \n\n 连接的段落字符串。
    """
    if not soup:
        return ""

    content_div = soup.find('div', class_='mw-parser-output')
    if not content_div:
        return ""

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

    paragraphs = []
    for element in content_div.find_all(['p', 'h2', 'h3']):
        text = element.get_text(separator=' ', strip=True)
        if len(text) >= 30:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def clean_blog(soup):
    """
    针对 WordPress 博客的提取逻辑。
    优先找 entry-content div，fallback 到所有 <p>。
    返回用 \n\n 连接的段落字符串。
    """
    if not soup:
        return ""

    content_div = soup.find('div', class_='entry-content')
    if not content_div:
        content_div = soup.find('article')
    if not content_div:
        content_div = soup

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
    【改进版】从 List_of_Asian_cuisines 提取东亚菜系链接。

    借鉴对方脚本的"动态层级刹车"逻辑：
      - 兼容维基新版 DOM：<div class="mw-heading"> 包裹标题的情况
      - 通过比较标题层级（h2/h3/h4 等）判断何时停止采集，
        遇到同级或更高级标题时刹车，遇到更低级标题时继续
      - 比原有 anchor 定位 + 关键词 fallback 更鲁棒

    同时保留原有的关键词 fallback 作为兜底。
    """
    base_url = "https://en.wikipedia.org/wiki/List_of_Asian_cuisines"
    soup = get_soup(base_url)
    links = set()
    if not soup:
        return list(links)

    content_div = soup.find('div', class_='mw-parser-output')
    if not content_div:
        return list(links)

    target_keywords = ["East Asian cuisine", "Chinese cuisine"]

    for keyword in target_keywords:
        # 在 h2~h6 中找到包含关键词的标题标签
        header_tag = content_div.find(
            lambda tag: tag.name in ['h2', 'h3', 'h4', 'h5', 'h6']
            and keyword.lower() in tag.text.lower()
        )
        if not header_tag:
            continue

        print(f"  [Wiki] 锁定大类: {keyword}，开始向下扫描...")

        # 【借鉴】处理新版维基 DOM：标题可能被包在 <div class="mw-heading"> 里
        # 如果是，要从父节点出发找兄弟，否则从标题本身出发
        if (header_tag.parent.name == 'div'
                and 'mw-heading' in header_tag.parent.get('class', [])):
            start_node = header_tag.parent
        else:
            start_node = header_tag

        # 【借鉴】记录当前标题层级，用于刹车判断
        current_level = int(header_tag.name[1])

        for sibling in start_node.find_next_siblings():
            # 【借鉴】判断兄弟节点是否为新标题，兼容新旧两种 DOM 结构
            is_heading = False
            heading_level = 99

            if sibling.name in ['h2', 'h3', 'h4', 'h5', 'h6']:
                is_heading = True
                heading_level = int(sibling.name[1])
            elif (sibling.name == 'div'
                  and 'mw-heading' in sibling.get('class', [])):
                is_heading = True
                inner_h = sibling.find(['h2', 'h3', 'h4', 'h5', 'h6'])
                if inner_h:
                    heading_level = int(inner_h.name[1])

            # 【借鉴】动态层级刹车：同级或更高级标题出现时停止
            # 更低级（如 h6 在 h5 下面）则继续采集
            if is_heading and heading_level <= current_level:
                break

            # 采集当前区块下的所有有效链接
            if sibling.name in ['ul', 'div', 'p']:
                for a in sibling.find_all('a'):
                    href = a.get('href', '')
                    if (href.startswith('/wiki/')
                            and ':' not in href
                            and 'Main_Page' not in href):
                        links.add(urljoin(base_url, href))

    # 原有关键词 fallback 保留：防止 anchor / DOM 结构变化时完全失效
    keywords = ['chinese', 'japanese', 'korean', 'taiwanese', 'hong_kong', 'macanese']
    for a in soup.find_all('a', href=True):
        href = a['href'].lower()
        if (href.startswith('/wiki/')
                and 'cuisine' in href
                and any(k in href for k in keywords)):
            full = urljoin(base_url, a['href'])
            links.add(full)

    return list(links)


def get_wikibooks_links():
    """
    【改进版】Wikibooks 链接收集。

    策略一（借鉴对方）：直接访问底层 Category 页，绕过前端折叠 UI。
      这是主策略，覆盖面最广、最可靠。

    策略二（原有）：从总目录页 + 硬编码入口做两层爬取。
      作为补充，捞回 Category 页可能遗漏的内容。

    两路结果合并去重。
    """
    wikibooks_base = "https://en.wikibooks.org"
    all_links = set(WIKIBOOKS_ENTRY_PAGES)

    # ── 策略一（借鉴）：直接访问底层 Category 页 ──
    print("  [Wikibooks] 策略一：扫描底层 Category 页...")
    for cat_url in WIKIBOOKS_CATEGORY_URLS:
        print(f"    扫描分类: {cat_url}")
        soup = get_soup(cat_url)
        if not soup:
            time.sleep(1)
            continue

        # 维基分类页的词条在 mw-category 或 mw-pages 这个 div 里
        category_div = (soup.find('div', class_='mw-category')
                        or soup.find('div', id='mw-pages'))
        if category_div:
            for a in category_div.find_all('a', href=True):
                href = a['href']
                if (href.startswith('/wiki/Cookbook:')
                        and 'Category:' not in href
                        and 'File:' not in href
                        and 'Special:' not in href):
                    all_links.add(urljoin(wikibooks_base, href))
        time.sleep(1)

    # ── 策略二（原有）：总目录页 + 硬编码入口两层爬取 ──
    print("  [Wikibooks] 策略二：两层爬取总目录页...")
    cuisine_pages = set(WIKIBOOKS_ENTRY_PAGES)

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

    for cuisine_url in cuisine_pages:
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
    同时用关键词动态匹配作为补充。
    """
    east_asian_keywords = ["China", "Japan", "South Korea", "North Korea",
                            "Taiwan", "Mongolia", "Macau", "Hong Kong"]
    links = set()

    # 主策略：直接遍历分类页
    for cat_url in BLOG_CATEGORY_URLS:
        print(f"  [Blog] 扫描分类: {cat_url}")
        soup = get_soup(cat_url)
        if not soup:
            time.sleep(1)
            continue
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'aroundtheworldin80cuisinesblog' in href.lower():
                links.add(href)
        time.sleep(1)

    # 备用策略：从首页侧边栏动态匹配关键词
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
    print(f"  Wikipedia (改进版层级刹车): {len(wiki_links)} links")

    wikibooks_links = get_wikibooks_links()
    print(f"  Wikibooks (Category页+两层爬取): {len(wikibooks_links)} links")

    blog_links = get_blog_links()
    print(f"  Blog:                      {len(blog_links)} links")

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