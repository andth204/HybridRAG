from typing import List
from pydantic import BaseModel, Field

class Route(BaseModel):
    name: str = Field(...)
    samples: List[str] = Field(default_factory=list)

    def add_sample(self, sample: str):
        if sample not in self.samples:
            self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)

__all__ = ["Route"]