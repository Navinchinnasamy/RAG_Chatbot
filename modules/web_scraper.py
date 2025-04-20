import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
from modules.config_loader import load_config
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
import hashlib

class WebScraper:
    def __init__(self, config):
        self.config = config
        self.base_domain = config["scraper"]["base_domain"]
        self.pages_config = config["scraper"]["pages"]
        self.articles_config = config["scraper"]["articles"]
        self.output_dir = config["scraper"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.content_hashes = set()

    def scrape_page(self, url):
        try:
            print(f"Started scraping {url}:")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove common header, footer, nav, and button elements
            for element in soup.find_all(['header', 'footer', 'nav', 'button', 'a']):
                element.decompose()

            # Exclude specific classes/IDs commonly used for menus/footers
            for element in soup.find_all(class_=['navbar', 'footer', 'menu', 'header', 'skip-to-content', 'profile-icon']):
                element.decompose()
            for element in soup.find_all(id=['navbar', 'footer', 'menu', 'header']):
                element.decompose()

            # Extract content from main/article or specific elements
            content = []
            # Try to find main or article content
            main_content = soup.find(['main', 'article'])
            if main_content:
                for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'td', 'th']):
                    text = element.get_text(separator=" ", strip=True)
                    if text:
                        content.append(text)
            else:
                # Fallback to paragraphs, headings, and tables
                for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'td', 'th']):
                    text = element.get_text(separator=" ", strip=True)
                    if text:
                        content.append(text)

            # Remove common noisy phrases
            noisy_phrases = ["skip to content", "register / login", "gold price gold", "profile icon", "my account"]
            cleaned_content = " ".join(content)
            for phrase in noisy_phrases:
                cleaned_content = cleaned_content.replace(phrase, "").strip()

            return cleaned_content if cleaned_content else None
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def save_content(self, content, filename):
        if content:
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            if content_hash in self.content_hashes:
                print(f"Skipping duplicate content for {filename}")
                return
            self.content_hashes.add(content_hash)
            safe_filename = filename.replace("/", "_").replace(":", "_")
            with open(os.path.join(self.output_dir, safe_filename), "w", encoding="utf-8") as f:
                f.write(content)

    def get_urls_from_sitemap(self, sitemap_url):
        try:
            response = requests.get(sitemap_url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            namespace = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            urls = [elem.text for elem in root.findall(".//sitemap:loc", namespace)]
            return urls
        except Exception as e:
            print(f"Error parsing sitemap {sitemap_url}: {e}")
            return []

    def get_urls_from_sitemaps(self, sitemap_urls):
        all_urls = []
        for sitemap_url in sitemap_urls:
            response = requests.get(sitemap_url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            namespace = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            if root.tag.endswith("sitemapindex"):
                sitemap_urls_inner = [elem.text for elem in root.findall(".//sitemap:loc", namespace)]
                for inner_sitemap_url in sitemap_urls_inner:
                    all_urls.extend(self.get_urls_from_sitemap(inner_sitemap_url))
            else:
                all_urls.extend(self.get_urls_from_sitemap(sitemap_url))
        return all_urls

    def scrape_urls(self, urls, prefix, max_items, desc):
        for i, url in enumerate(tqdm(urls[:max_items], desc=desc)):
            if prefix == "article" and "articles" in self.articles_config.get("url_pattern", "") and "articles" not in url:
                print(f"Skipping {url}: Does not match expected article pattern.")
                continue
            content = self.scrape_page(url)
            if content:
                filename = f"{prefix}_{url.replace(self.base_domain, '').replace('/', '_')}.txt"
                self.save_content(content, filename)

    def scrape(self):
        page_urls = []
        if self.pages_config.get("sitemaps"):
            page_urls = self.get_urls_from_sitemaps(self.pages_config["sitemaps"])
        elif self.pages_config.get("urls"):
            page_urls = self.pages_config["urls"]
        elif self.pages_config.get("url_pattern"):
            page_urls = [
                self.pages_config["url_pattern"].format(i)
                for i in range(1, self.pages_config["max_pages"] + 1)
            ]
        else:
            print("Error: No sitemaps, urls, or url_pattern defined for pages.")
            return

        self.scrape_urls(page_urls, "page", self.pages_config["max_pages"], "Scraping pages")

        article_urls = []
        if self.articles_config.get("sitemaps"):
            article_urls = self.get_urls_from_sitemaps(self.articles_config["sitemaps"])
        elif self.articles_config.get("urls"):
            article_urls = self.articles_config["urls"]
        elif self.articles_config.get("url_pattern"):
            article_urls = [
                self.articles_config["url_pattern"].format(i)
                for i in range(1, self.articles_config["max_articles"] + 1)
            ]
        else:
            print("Error: No sitemaps, urls, or url_pattern defined for articles.")
            return

        self.scrape_urls(article_urls, "article", self.articles_config["max_articles"], "Scraping articles")

if __name__ == "__main__":
    config = load_config()
    scraper = WebScraper(config)
    scraper.scrape()