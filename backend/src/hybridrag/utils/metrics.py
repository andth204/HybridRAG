import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    return len(tiktoken.encoding_for_model(model).encode(text))

def truncate_text(text: str, max_tokens: int, model: str = "gpt-4o-mini") -> str:
    tokens = tiktoken.encoding_for_model(model).encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tiktoken.encoding_for_model(model).decode(tokens[:max_tokens])

__all__ = [
    "count_tokens",
    "truncate_text"
]