from pathlib import Path
from functools import lru_cache
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from src.config.settings import settings

class TextSplitter:
    def __init__(self, 
                 chunk_size: int = settings.FIXED_CHUNK_SIZE, 
                 chunk_overlap: int = settings.FIXED_CHUNK_OVERLAP, 
                 encoding_name: str = "cl100k_base",
                 header_tokens: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        self.header_tokens = header_tokens
        
    @lru_cache(maxsize=1)
    def _load_encoder(self):
        return tiktoken.get_encoding(self.encoding_name)

    def get_splitter(self) -> RecursiveCharacterTextSplitter:
        encoder = self._load_encoder()
        effective_chunk_size = self.chunk_size - self.header_tokens
        return RecursiveCharacterTextSplitter(
            chunk_size=effective_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=lambda x: len(encoder.encode(x)),
        )
    
    def split_text(self, text: str) -> list:
        encoder = self._load_encoder()
        tokens = encoder.encode(text)
        header_tokens = tokens[:self.header_tokens]
        header_text = encoder.decode(header_tokens)
        initial_chunks = self.get_splitter().split_text(text)
        optimized_chunks = self._optimize_chunks(initial_chunks, header_text)
        return optimized_chunks
    
    def _optimize_chunks(self, chunks: list, header_text: str) -> list:
        if len(chunks) <= 1:
            return chunks
        encoder = self._load_encoder()
        optimized = [chunks[0]]
        for i in range(1, len(chunks)):
            current_chunk = optimized[-1]
            next_chunk = chunks[i]
            combined_text = current_chunk + "\n\n" + header_text + "\n\n" + "...\n\n" + next_chunk
            combined_tokens = len(encoder.encode(combined_text))
            if combined_tokens <= self.chunk_size:
                optimized[-1] = combined_text
            else:
                optimized.append(header_text + "\n\n" + "...\n\n" + next_chunk)
        return optimized

def main():
    text_splitter = TextSplitter()
    file_path = Path(r"D:/python-projects/nlp/generation/SmartChat/backend/data/raw/utehy_edu_vn.md")
    
    with open(file_path, 'r', encoding='utf-8') as file:    
        content = file.read()
    
    chunks = text_splitter.split_text(content)
    encoder = text_splitter._load_encoder()
    
    print(f"Original text: {len(content)} chars, {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        token_count = len(encoder.encode(chunk))
        print(f"Chunk {i+1}: {len(chunk)} chars, {token_count} tokens")
        print(f"Content: {chunk[:500]}{'...' if len(chunk) > 500 else ''}\n")

if __name__ == "__main__":
    main()