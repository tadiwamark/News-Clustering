# -*- coding: utf-8 -*-
"""dailynews_spider.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GBzpMxdY_tFgLFZ6N6MaipLdu7bR3D6q
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

class DailyNewsSpider:
    def __init__(self):
        """Initializes the spider with a requests session."""
        self.session = requests.Session()

    def scrape_dailynews_articles(self, url, category):
        """Scrapes articles from the given URL under the specified category."""
        response = self.session.get(url)
        response.raise_for_status()  # Ensure the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = []
        post_containers = soup.find_all('div', class_='blog-post inhype-block inhype-standard-post')
        for post in post_containers:
            title_link = post.find('h3', class_='post-title entry-title').find('a')
            if title_link:
                title = title_link.get_text(strip=True)
                url = title_link['href']
                story_snippet = post.find('div', class_='post-excerpt').p.get_text(strip=True) if post.find('div', class_='post-excerpt').p else ''
                articles.append({
                    'category': category,
                    'title': title,
                    'story': story_snippet,
                    'url': url
                })

        return articles

    def run(self):
        """Main execution method to scrape all categories."""
        categories_urls = {
            'Business': ['https://dailynews.co.zw/business/', 'https://dailynews.co.zw/?s=business', 'https://dailynews.co.zw/page/2/?s=business', 'https://dailynews.co.zw/page/3/?s=business', 'https://dailynews.co.zw/page/4/?s=business'],
            'Politics': ['https://dailynews.co.zw/?s=zanu'],
            'Arts & Culture': ['https://dailynews.co.zw/?s=musician', 'https://dailynews.co.zw/?s=arts', 'https://dailynews.co.zw/?s=culture'],
            'Sports': ['https://dailynews.co.zw/sport/', 'https://dailynews.co.zw/?s=sport', 'https://dailynews.co.zw/page/2/?s=sport','https://dailynews.co.zw/page/3/?s=sport', 'https://dailynews.co.zw/page/4/?s=sport']
        }

        all_articles = []
        for category, urls in categories_urls.items():
            for url in urls:
                print(f"Scraping {category} articles from {url}...")
                all_articles += self.scrape_dailynews_articles(url, category)

        # Convert the list of articles to a DataFrame and save it
        df_articles = pd.DataFrame(all_articles)
        print(df_articles)
        df_articles.to_csv('dailynews_articles_comb.csv', index=False)


spider = DailyNewsSpider()
spider.run()

