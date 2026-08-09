from pydantic import BaseModel, Field


class ProductOption(BaseModel):
    value: str
    price: float | None = None
    is_available: bool = True
    asin: str | None = None

class ProductVariant(BaseModel):
    options: dict[str, str]
    price: float | None = None
    asin: str | None = None
    is_available: bool = True


class ProductDetail(BaseModel):
    asin: str
    title: str
    brand: str | None = None
    category: str
    price_min: float | None = None
    price_max: float | None = None
    pricing: list[float] = Field(default_factory=list)
    description: str = ""
    attributes: list[str] = Field(default_factory=list)
    customization_options: dict[str, list[ProductOption]] = Field(default_factory=dict)
    variant_combinations: list[ProductVariant] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)

class ProductSummary(BaseModel):
    asin: str
    title: str
    category: str
    brand: str | None = None
    price_min: float | None = None
    price_max: float | None = None
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