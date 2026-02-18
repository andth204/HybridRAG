from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_reranker(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return tokenizer, model

def count_parameters(model) -> int:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters    : {total:,}")
    print(f"   Trainable parameters: {trainable:,}")
    return total

def rerank(query: str, documents: list[str], tokenizer, model, k: int = 3) -> list[dict]:
    pairs = [[query, doc] for doc in documents]
    
    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        scores = model(**inputs).logits.squeeze(-1).float()
        scores = torch.sigmoid(scores).tolist()  # normalize về [0, 1]
    results = [{"score": score, "document": doc} for score, doc in zip(scores, documents)]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:k]


if __name__ == "__main__":
    MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
    tokenizer, model = load_reranker(MODEL_NAME)
    count_parameters(model)
    
    print(f"\n=> Loading model: {MODEL_NAME}")
    query = "Triệu chứng của bệnh tiểu đường là gì?"
    documents = [
        "Bệnh tiểu đường gây ra triệu chứng khát nước nhiều, tiểu nhiều lần.",
        "Hà Nội là thủ đô của Việt Nam.",
        "Insulin giúp điều hòa lượng đường trong máu.",
        "Mèo là loài động vật phổ biến.",
        "Bệnh nhân tiểu đường thường cảm thấy mệt mỏi và sụt cân không rõ nguyên nhân.",
        "Python là ngôn ngữ lập trình phổ biến.",
        "Chế độ ăn ít đường giúp kiểm soát bệnh tiểu đường type 2.",
        "Hôm nay thời tiết đẹp.",
        "Xét nghiệm HbA1c dùng để chẩn đoán và theo dõi bệnh tiểu đường.",
        "Finetuning mô hình ngôn ngữ lớn đòi hỏi nhiều GPU.",
    ]

    print(f"\n=> Query: {query}")
    print(f"=> Reranking {len(documents)} documents, lấy top k=3...\n")

    top_k = rerank(query, documents, tokenizer, model, k=3)
    for i, item in enumerate(top_k, 1):
        print(f"   [{i}] Score: {item['score']:.4f} | {item['document']}")