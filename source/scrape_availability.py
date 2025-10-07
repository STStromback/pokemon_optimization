#!/usr/bin/env python3
"""
Pokemon Location Data Scraper
Scrapes Pokemon location data from pokemondb.net for Red, Crystal, and Emerald versions.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import urljoin, urlparse
import logging
import os
import json
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PokemonLocationScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = "https://pokemondb.net"
        self.crawl_delay = 2  # Respect robots.txt
        
        # URL mappings for each generation (all use main page with different regions)
        self.generation_urls = {
            "1": "https://pokemondb.net/location",
            "2": "https://pokemondb.net/location", 
            "3": "https://pokemondb.net/location"
        }
        
        # Game version mappings
        self.target_versions = {
            "1": "R",  # Red
            "2": "C",  # Crystal
            "3": "E"   # Emerald
        }
        
        self.results = []
        
        # Method mappings for h3 tag content
        self.method_mappings = {
            'surfing': 'surf',
            'super rod': 'super_rod',
            'good rod': 'good_rod', 
            'old rod': 'old_rod',
            'rock smash': 'rock_smash',
            'headbutt': 'headbutt',
            'headbutt (special)': 'headbutt'
        }
        


    def get_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a webpage with crawl delay."""
        logger.info(f"Fetching: {url}")
        time.sleep(self.crawl_delay)
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def extract_location_links(self, soup: BeautifulSoup, generation: str) -> List[Tuple[str, str]]:
        """Extract location links from the main generation page."""
        links = []
        
        # Region mappings for each generation
        # Gen 2 includes both Johto and Kanto since Crystal has postgame Kanto content
        region_prefixes = {
            "1": ["kanto"],           # Generation 1 = Kanto only
            "2": ["johto", "kanto"],  # Generation 2 = Johto + Kanto (Crystal postgame)
            "3": ["hoenn"]            # Generation 3 = Hoenn only
        }
        
        target_regions = region_prefixes.get(generation, [])
        if not target_regions:
            return links
        
        # Find all links that contain any of our target region prefixes in their href
        all_links = soup.find_all('a', href=True)
        
        for region_prefix in target_regions:
            for link in all_links:
                href = link.get('href', '')
                # Check if this link is for a location in our target region
                if f'/location/{region_prefix}-' in href and href.startswith('/location/'):
                    location_name = link.text.strip()
                    if location_name:  # Make sure we have a valid name
                        location_url = urljoin(self.base_url, href)
                        links.append((location_name, location_url))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for name, url in links:
            if url not in seen:
                seen.add(url)
                unique_links.append((name, url))
        
        logger.info(f"Found {len(unique_links)} location links for generation {generation} ({', '.join(target_regions)})")
        return unique_links

    def parse_pokemon_table(self, soup: BeautifulSoup, location_name: str, generation: str) -> List[Dict]:
        """Parse Pokemon data from a location page."""
        pokemon_data = []
        target_version = self.target_versions[generation]
        
        # First try the table format (for newer layout)
        tables = soup.find_all('table', {'class': 'data-table'})
        
        for table in tables:
            tbody = table.find('tbody')
            if not tbody:
                continue
                
            # Find the sub_location from the nearest h2 tag above this table
            sub_location = self._get_sub_location(table, generation)
                
            rows = tbody.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 6:  # Need at least 6 cells for expected format
                    continue
                
                # Extract Pokemon name
                name_cell = cells[0]
                pokemon_link = name_cell.find('a', {'class': 'ent-name'})
                if not pokemon_link:
                    continue
                pokemon_name = pokemon_link.text.strip()
                
                # Check game version cells (typically cells 1-3)
                version_found = False
                for i in range(1, min(4, len(cells))):
                    cell = cells[i]
                    cell_text = cell.text.strip()
                    cell_classes = cell.get('class', [])
                    
                    # Check if this cell represents our target version and is not blank
                    if cell_text == target_version:
                        # Check if any class contains 'blank' - if so, this Pokemon is not available
                        has_blank = any('blank' in cls for cls in cell_classes)
                        if not has_blank:
                            # Also verify it has the target game class (not just blank)
                            has_target = any(f'-{target_version}' in cls for cls in cell_classes)
                            if has_target:
                                version_found = True
                                break
                
                if not version_found:
                    continue
                
                # Extract levels (usually in the 5th or 6th cell)
                levels = ""
                for i in range(4, len(cells)):
                    cell = cells[i]
                    if 'cell-num' in cell.get('class', []) or re.search(r'\d+-?\d*', cell.text):
                        levels = cell.text.strip()
                        break
                
                if levels:
                    # Find the current method (h3 tag above the table)
                    method = self._get_current_method(table)
                    pokemon_data.append({
                        'gen': generation,
                        'location': location_name,
                        'sub_location': sub_location,
                        'pokemon': pokemon_name,
                        'levels': levels,
                        'method': method
                    })
        
        # Only process non-table data if we didn't find any tables
        # This avoids duplicate entries and spurious 'walking' entries
        if not tables:
            # Look for ALL generation headings for this generation
            gen_headings = soup.find_all('h2')
            
            for heading in gen_headings:
                if f'Generation {generation}' in heading.text:
                    # Extract sub_location from h2 text
                    sub_location = self._extract_sub_location_from_text(heading.text, generation)
                    
                    # Found our generation section, look for Pokemon links after it
                    current_element = heading.find_next_sibling()
                    
                    while current_element and current_element.name != 'h2':
                        if current_element.name == 'h3':  # Walking, Surfing, etc.
                            # Look for Pokemon links in the next elements
                            method_element = current_element.find_next_sibling()
                            while method_element and method_element.name not in ['h2', 'h3']:
                                # Find Pokemon links
                                pokemon_links = method_element.find_all('a', href=True)
                                for link in pokemon_links:
                                    href = link.get('href', '')
                                    if '/pokedex/' in href and href != '/pokedex/':
                                        pokemon_name = link.text.strip()
                                        if pokemon_name:
                                            method = current_element.text.strip() if current_element.name == 'h3' else 'varies'
                                            method = self._normalize_method(method)
                                            pokemon_data.append({
                                                'gen': generation,
                                                'location': location_name,
                                                'sub_location': sub_location,
                                                'pokemon': pokemon_name,
                                                'levels': 'varies',  # Default when level info not available
                                                'method': method
                                            })
                                
                                method_element = method_element.find_next_sibling()
                        
                        current_element = current_element.find_next_sibling()
                    
                    # Continue processing all h2 sections for this generation
        
        return pokemon_data
    
    def _get_current_method(self, table_element) -> str:
        """Find the h3 method tag above the current table."""
        # First try to find h3 among direct siblings
        current = table_element
        while current:
            current = current.find_previous_sibling()
            if current and current.name == 'h3':
                method_text = current.text.strip()
                return self._normalize_method(method_text)
            elif current and current.name == 'h2':
                break  # Stop if we hit another generation heading
        
        # If not found among siblings, traverse up through parent containers
        # and look for the nearest h3 that comes before this table in document order
        parent = table_element.parent
        while parent and parent.name != 'body':
            # Look for previous siblings of parent that might contain h3
            prev = parent.find_previous_sibling()
            while prev:
                # Check if this element is or contains an h3
                if prev.name == 'h3':
                    method_text = prev.text.strip()
                    return self._normalize_method(method_text)
                # Also check for h3 within this element
                h3 = prev.find('h3')
                if h3:
                    # Make sure this h3 comes before our table
                    method_text = h3.text.strip()
                    return self._normalize_method(method_text)
                if prev.name == 'h2':
                    break
                prev = prev.find_previous_sibling()
            
            # Move up to parent's parent
            parent = parent.parent
            
        return 'walking'  # Default method
    
    def _get_sub_location(self, table_element, generation: str) -> str:
        """Find the sub_location from the nearest h2 tag above the current table."""
        # Look for h2 tags that contain "Generation X" to find sub_location
        # Start from the table and work backwards through the document
        current = table_element
        
        # First try direct traversal up the DOM
        while current:
            # Check all previous siblings
            sibling = current.find_previous_sibling()
            while sibling:
                if sibling.name == 'h2' and f'Generation {generation}' in sibling.text:
                    return self._extract_sub_location_from_text(sibling.text, generation)
                sibling = sibling.find_previous_sibling()
            
            # Move up to parent
            current = current.parent
            if not current or current.name == 'body':
                break
        
        # If still not found, search the entire document for relevant h2 tags
        # This handles cases where the h2 is in a different container
        soup = table_element.find_parent('html') or table_element.find_parent()
        if soup:
            gen_h2s = soup.find_all('h2')
            # Find the closest h2 with our generation that comes before this table
            table_pos = self._get_element_position(table_element)
            best_h2 = None
            best_pos = -1
            
            for h2 in gen_h2s:
                if f'Generation {generation}' in h2.text:
                    h2_pos = self._get_element_position(h2)
                    if h2_pos < table_pos and h2_pos > best_pos:
                        best_h2 = h2
                        best_pos = h2_pos
            
            if best_h2:
                return self._extract_sub_location_from_text(best_h2.text, generation)
        
        return ''  # Default empty sub_location
    
    def _get_element_position(self, element):
        """Get the position of an element in the document for ordering."""
        position = 0
        for elem in element.find_parent().find_all():
            if elem == element:
                break
            position += 1
        return position
    
    def _extract_sub_location_from_text(self, text: str, generation: str) -> str:
        """Extract sub_location from h2 text after the dash."""
        # Look for pattern like "Generation 3 - Outside" and extract "Outside"
        pattern = f'Generation {generation}\s*-\s*(.+)'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return ''  # Default empty sub_location
    
    def _normalize_method(self, method_text: str) -> str:
        """Normalize method text according to specified mappings."""
        method_lower = method_text.lower().strip()
        return self.method_mappings.get(method_lower, method_lower)

    def scrape_location(self, location_name: str, location_url: str, generation: str) -> List[Dict]:
        """Scrape Pokemon data from a specific location page."""
        logger.info(f"Scraping location: {location_name} (Gen {generation})")
        
        soup = self.get_page(location_url)
        if not soup:
            return []
        
        return self.parse_pokemon_table(soup, location_name, generation)

    def scrape_all_generations(self):
        """Main scraping function for all generations."""
        logger.info("Starting Pokemon location data scraping...")
        
        for generation, main_url in self.generation_urls.items():
            logger.info(f"Processing Generation {generation}")
            
            # Get main page for this generation
            soup = self.get_page(main_url)
            if not soup:
                logger.error(f"Failed to load main page for generation {generation}")
                continue
            
            # Extract location links
            location_links = self.extract_location_links(soup, generation)
            
            # Scrape each location
            for location_name, location_url in location_links:
                try:
                    pokemon_data = self.scrape_location(location_name, location_url, generation)
                    self.results.extend(pokemon_data)
                    logger.info(f"Found {len(pokemon_data)} Pokemon at {location_name}")
                except Exception as e:
                    logger.error(f"Error scraping {location_name}: {e}")
                    continue
        
        logger.info(f"Scraping completed. Total records: {len(self.results)}")

    def save_raw_to_csv(self, output_path: str = "pokemon_availability_raw.csv"):
        """Save raw scraping results to CSV file without processing."""
        if not self.results:
            logger.warning("No data to save")
            return
        
        df = pd.DataFrame(self.results)
        df = df.drop_duplicates().sort_values(['gen', 'location', 'pokemon'])
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Raw data saved to {output_path}")
        logger.info(f"Total unique records: {len(df)}")
        
        # Print summary statistics
        print("\n=== SCRAPING SUMMARY ===")
        print(f"Total records: {len(df)}")
        for gen in sorted(df['gen'].unique()):
            gen_data = df[df['gen'] == gen]
            version_name = {'1': 'Red', '2': 'Crystal', '3': 'Emerald'}[gen]
            print(f"Generation {gen} ({version_name}): {len(gen_data)} records")
            print(f"  Unique Pokemon: {gen_data['pokemon'].nunique()}")
            print(f"  Unique Locations: {gen_data['location'].nunique()}")
            if 'method' in gen_data.columns:
                print(f"  Unique Methods: {gen_data['method'].nunique()}")

def main():
    """Main execution function."""
    scraper = PokemonLocationScraper()
    
    try:
        scraper.scrape_all_generations()
        
        # Save raw data to the gen_all directory
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_path = os.path.join(base_path, 'data', 'gen_all', 'pokemon_availability_raw.csv')
        scraper.save_raw_to_csv(output_path)
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()
