import re
from urllib.parse import urlparse
from app.features.detection.schemas import URLFeatureValues


class URLExtractor:
    """Extractor for deriving 30 phishing detection features from a raw URL string."""

    SHORTENING_SERVICES = r"(bit\.ly|tinyurl\.com|goo\.gl|ow\.ly|is\.gd|buff\.ly|adf\.ly|t\.co|lnkd\.in)"

    @classmethod
    def extract_features(cls, url: str) -> URLFeatureValues:
        """Parse URL and map characteristics to feature dictionary."""
        # Ensure url has scheme for parsing
        if not re.match(r"^https?://", url, re.IGNORECASE):
            raw_url = "http://" + url
        else:
            raw_url = url

        parsed = urlparse(raw_url)
        hostname = parsed.hostname or ""
        path = parsed.path or ""

        # 1. having_IP_Address
        ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
        has_ip = 1 if re.match(ip_pattern, hostname) else -1

        # 2. URL_Length
        if len(url) < 54:
            url_len = -1
        elif 54 <= len(url) <= 75:
            url_len = 0
        else:
            url_len = 1

        # 3. Shortining_Service
        short_service = 1 if re.search(cls.SHORTENING_SERVICES, hostname, re.IGNORECASE) else -1

        # 4. having_At_Symbol
        at_symbol = 1 if "@" in url else -1

        # 5. double_slash_redirecting
        double_slash = 1 if raw_url.rfind("//") > 7 else -1

        # 6. Prefix_Suffix
        prefix_suffix = 1 if "-" in hostname else -1

        # 7. having_Sub_Domain
        dots_count = hostname.count(".")
        if dots_count <= 1:
            subdomain = -1
        elif dots_count == 2:
            subdomain = 0
        else:
            subdomain = 1

        # 8. SSLfinal_State
        is_https = parsed.scheme.lower() == "https"
        ssl_state = -1 if is_https else 1

        # 11. port
        port_num = parsed.port
        has_custom_port = 1 if port_num and port_num not in (80, 443) else -1

        # 12. HTTPS_token
        https_token = 1 if "https" in hostname.lower() else -1

        return URLFeatureValues(
            having_IP_Address=has_ip,
            URL_Length=url_len,
            Shortining_Service=short_service,
            having_At_Symbol=at_symbol,
            double_slash_redirecting=double_slash,
            Prefix_Suffix=prefix_suffix,
            having_Sub_Domain=subdomain,
            SSLfinal_State=ssl_state,
            Domain_registeration_length=-1,
            Favicon=-1,
            port=has_custom_port,
            HTTPS_token=https_token,
            Request_URL=-1,
            URL_of_Anchor=0,
            Links_in_tags=0,
            SFH=-1,
            Submitting_to_email=-1,
            Abnormal_URL=-1,
            Redirect=0,
            on_mouseover=-1,
            RightClick=-1,
            popUpWidnow=-1,
            Iframe=-1,
            age_of_domain=-1,
            DNSRecord=-1,
            web_traffic=0,
            Page_Rank=-1,
            Google_Index=-1,
            Links_pointing_to_page=0,
            Statistical_report=-1,
        )
