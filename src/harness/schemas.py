from pydantic import BaseModel, Field


class ProductSummary(BaseModel):
    asin: str
    title: str
    category: str
    brand: str | None = None
    price_min: float | None = None
    bm25_score: float
    rank: int


class SearchProductsAction(BaseModel):
    query: str
    limit: int = Field(default=20, ge=1, le=50)


class GetProductAction(BaseModel):
    asin: str


class RecommendationItem(BaseModel):
    asin: str
    selected_options: dict[str, str]
    reason: str


class SubmitRecommendationsAction(BaseModel):
    recommendations: list[RecommendationItem]