"""
MAPA Wine Price Scraper - Production Version
Target: Weekly bulk wine prices (White/Red) from MAPA PDF bulletins.
Output: CSV dataset for RNN ingestion pipeline.
"""

import requests
import pdfplumber
import pandas as pd
import re
import io
import time
import logging
from typing import Optional, List, Dict, Union
from datetime import datetime
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wine_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WinePriceScraper")

class MAPAWineScraper:
    """
    Robust scraper for extracting wine prices from MAPA bulletins.
    Implements retry logic, PDF table parsing, and structured logging.
    """

    BASE_URL = "https://www.mapa.gob.es"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (DataPipeline/1.0; WinePriceForecasting)'
    }

    def __init__(self):
        """Initialize the scraper session with retry strategy for network resilience."""
        self.session = requests.Session()

        # Retry strategy: 3 retries with exponential backoff for 50x errors
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Dynamically generate campaign URLs (current + recent history)
        # Covers the requirements for historical training data [cite: 42]
        self.campaigns = self._generate_campaign_config(start_year=2022)
        self.collected_data = []

    def _generate_campaign_config(self, start_year: int) -> Dict[str, str]:
        """
        Generates campaign URLs dynamically to ensure scalability for future seasons.
        Format: /.../boletines_semanales_precio_vino_YYYY_YYYY+1
        """
        current_year = datetime.now().year + 1
        campaigns = {}

        for year in range(start_year, current_year):
            cycle = f"{year}_{year+1}"
            label = f"{year}/{year+1}"
            # URL path based on MAPA's standard structure
            url_path = (
                f"/es/agricultura/temas/producciones-agricolas/vitivinicultura/"
                f"boletines_semanales_precio_vino/boletines_semanales_precio_vino_{cycle}"
            )
            campaigns[label] = url_path

        return campaigns

    def _fetch_url(self, url: str) -> Optional[bytes]:
        """Helper method to fetch URL content with error handling."""
        try:
            full_url = urljoin(self.BASE_URL, url) if not url.startswith("http") else url
            response = self.session.get(full_url, headers=self.HEADERS, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching {url}: {e}")
            return None

    def get_pdf_links(self, campaign_url: str) -> List[Dict[str, str]]:
        """Parses the campaign page to find all PDF bulletin links."""
        content = self._fetch_url(campaign_url)
        if not content:
            return []

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        pdf_links = []
        seen_urls = set()

        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)

            # Filter for PDFs containing 'boletin' in the URL
            if href.lower().endswith('.pdf') and 'boletin' in href.lower():
                full_url = urljoin(self.BASE_URL, href)

                if full_url not in seen_urls:
                    pdf_links.append({
                        'url': full_url,
                        'text': text
                    })
                    seen_urls.add(full_url)

        return pdf_links

    def extract_prices_from_pdf(self, pdf_bytes: bytes) -> Dict[str, Optional[float]]:
        """
        Extracts white and red wine prices using spatial layout analysis (pdfplumber).
        More robust than simple text stream reading.
        """
        prices = {'blanco': None, 'tinto': None}

        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                full_text = ""
                # Aggregate text from all pages
                for page in pdf.pages:
                    # Extract text preserving rough layout
                    text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    full_text += text + "\n"

            # Normalize decimal separators (European comma to dot)
            full_text_normalized = full_text.replace(',', '.')

            # Regex strategy: Look for 'Blanco'/'Tinto' followed by a price format
            # Matches: "Vino Blanco ...... 35.40" or "Blanco sin DOP: 35.40"
            # It captures the first valid price found after the keyword

            # 1. White Wine (Blanco)
            white_match = re.search(
                r'(?:blanco|mesa\s+blanco)[^0-9\n]*?(\d{2,3}\.\d{2})',
                full_text_normalized,
                re.IGNORECASE | re.DOTALL
            )
            if white_match:
                prices['blanco'] = float(white_match.group(1))

            # 2. Red Wine (Tinto)
            red_match = re.search(
                r'(?:tinto|mesa\s+tinto)[^0-9\n]*?(\d{2,3}\.\d{2})',
                full_text_normalized,
                re.IGNORECASE | re.DOTALL
            )
            if red_match:
                prices['tinto'] = float(red_match.group(1))

        except Exception as e:
            logger.error(f"Error parsing PDF content: {e}")

        return prices

    def process_campaign(self, campaign_name: str, url: str):
        """Process all bulletins within a specific campaign."""
        logger.info(f"Starting campaign processing: {campaign_name}")

        pdf_links = self.get_pdf_links(url)
        logger.info(f"Found {len(pdf_links)} bulletins.")

        for idx, link in enumerate(pdf_links):
            # Extract week number from link text (e.g., "Boletín semana 42")
            week_match = re.search(r'(\d+)', link['text'])
            week_num = int(week_match.group(1)) if week_match else 0

            # Download PDF
            pdf_bytes = self._fetch_url(link['url'])
            if not pdf_bytes:
                continue

            # Extract Prices
            prices = self.extract_prices_from_pdf(pdf_bytes)

            record = {
                'campaign': campaign_name,
                'week': week_num,
                'price_white': prices['blanco'],
                'price_red': prices['tinto'],
                'source_url': link['url'],
                'scraped_at': datetime.now().isoformat()
            }

            # Only append if at least one price was found or if we need to track missing data
            # Keeping None values is important for the interpolation step later [cite: 31]
            self.collected_data.append(record)

            logger.info(f"Processed Week {week_num}: White={prices['blanco']}, Red={prices['tinto']}")

            # Politeness delay
            time.sleep(1)

    def run(self):
        """Main execution method."""
        logger.info("Starting global scraping job...")

        for name, url in self.campaigns.items():
            try:
                self.process_campaign(name, url)
            except Exception as e:
                logger.error(f"Critical error in campaign {name}: {e}")

        if self.collected_data:
            df = pd.DataFrame(self.collected_data)

            # Sort by Campaign and Week for chronological order
            df.sort_values(by=['campaign', 'week'], inplace=True)

            filename = 'mapa_wine_prices_raw.csv'
            df.to_csv(filename, index=False, encoding='utf-8')

            logger.info(f"Job finished. Data saved to {filename}")
            logger.info(f"Total records: {len(df)}")

            # Summary statistics for validation
            logger.info(f"Valid White Prices: {df['price_white'].notna().sum()}")
            logger.info(f"Valid Red Prices: {df['price_red'].notna().sum()}")

            return df
        else:
            logger.warning("No data collected.")
            return None

if __name__ == "__main__":
    scraper = MAPAWineScraper()
    scraper.run()
