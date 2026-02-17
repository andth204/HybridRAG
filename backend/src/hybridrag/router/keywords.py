import re
import unicodedata
from typing import Tuple, List
from .route import Route

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return text.strip()

class KeywordRouter:
    def __init__(self, routes: List[Route]):
        self.routes = routes
        self.keyword_map = {}
        for route in routes:
            keywords = set()
            for sample in route.samples:
                normalized = normalize_text(sample)
                words = re.findall(r'\b\w+\b', normalized)
                keywords.update(words)
            self.keyword_map[route.name] = keywords

    def guide(self, query: str) -> Tuple[int, str]:
        normalized_query = normalize_text(query)
        query_words = set(re.findall(r'\b\w+\b', normalized_query))
        best_match_count = 0
        best_route = "unknown"
        for route_name, keywords in self.keyword_map.items():
            match_count = len(query_words & keywords)
            if match_count > best_match_count:
                best_match_count = match_count
                best_route = route_name
        return best_match_count, best_route

    def get_routes(self) -> List[Route]:
        return self.routes


__all__ = ["KeywordRouter"]