from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class URLInputRequest(BaseModel):
    """Request model for URL feature extraction."""
    url: str = Field(..., description="Target URL to be analyzed", examples=["https://secure-login.bank-verify.xyz"])


class URLFeatureValues(BaseModel):
    """The 30 feature attributes mapped from the URL."""
    having_IP_Address: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    URL_Length: int = Field(-1, description="-1: Normal, 0: Moderate, 1: Phishing")
    Shortining_Service: int = Field(-1, description="-1: Legitimate, 1: Shortened")
    having_At_Symbol: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    double_slash_redirecting: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    Prefix_Suffix: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    having_Sub_Domain: int = Field(-1, description="-1: Legitimate, 0: Moderate, 1: Phishing")
    SSLfinal_State: int = Field(-1, description="-1: Valid SSL, 0: Suspicious, 1: Phishing")
    Domain_registeration_length: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    Favicon: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    port: int = Field(-1, description="-1: Normal Port, 1: Phishing")
    HTTPS_token: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    Request_URL: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    URL_of_Anchor: int = Field(-1, description="-1: Legitimate, 0: Moderate, 1: Phishing")
    Links_in_tags: int = Field(-1, description="-1: Legitimate, 0: Moderate, 1: Phishing")
    SFH: int = Field(-1, description="-1: Legitimate, 0: Moderate, 1: Phishing")
    Submitting_to_email: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    Abnormal_URL: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    Redirect: int = Field(0, description="0: Legitimate, 1: Phishing")
    on_mouseover: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    RightClick: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    popUpWidnow: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    Iframe: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    age_of_domain: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    DNSRecord: int = Field(-1, description="-1: Legitimate, 1: Phishing")
    web_traffic: int = Field(-1, description="-1: High Traffic, 0: Moderate, 1: Phishing")
    Page_Rank: int = Field(-1, description="-1: High Rank, 1: Low/Phishing")
    Google_Index: int = Field(-1, description="-1: Indexed, 1: Phishing")
    Links_pointing_to_page: int = Field(-1, description="-1: High Links, 0: Moderate, 1: Phishing")
    Statistical_report: int = Field(-1, description="-1: Legitimate, 1: Phishing")


class ExtractedFeaturesResponse(BaseModel):
    """Response containing extracted 30 features."""
    url: str
    features: URLFeatureValues


class PredictionRequest(BaseModel):
    """Prediction request payload."""
    model_name: Optional[str] = Field("random_forest", description="Model algorithm to use: random_forest, decision_tree, knn")
    features: URLFeatureValues


class PredictionResponse(BaseModel):
    """Prediction response containing evaluation result."""
    prediction: str = Field(..., description="'Phishing' or 'Legitimate'")
    label: int = Field(..., description="1 for Phishing, -1 for Legitimate")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    model_used: str
    risk_factors: List[str] = Field(default_factory=list, description="List of suspicious features triggered")
