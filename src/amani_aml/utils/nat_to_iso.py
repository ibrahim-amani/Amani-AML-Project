"""
src.sanction_parser.scrapers.utils.nat_to_iso
Utility to map country names and demonyms to ISO 3166-1 alpha-2 country codes.
"""
from __future__ import annotations
import re
from typing import Any

_SPLIT_RE = re.compile(r"\s*(?:;|,|/|\||&|\+|\band\b|\bAND\b)\s*", flags=re.IGNORECASE)

COUNTRY_TO_ISO2: dict[str, str | list[str] | None] = {
    # --- Core / English ---
    "Pakistan": "PK",
    "Pakistani": "PK",
    "Afghan": "AF",
    "Afghani": "AF",
    "Afghanistan": "AF",
    "Afghanistan - NDE": "AF",
    "Afghanistan, Pakistan": ["AF", "PK"],
    "Albania": "AL",
    "Algeria": "DZ",
    "Algerian": "DZ",
    "Algeria, Australia": ["DZ", "AU"],
    "Algeria, Palestine": ["DZ", "PS"],
    "Andorra": "AD",
    "Angola": "AO",
    "Antigua and Barbuda": "AG",
    "Argentina": "AR",
    "Argentina - Non ADB Member Country": "AR",
    "Armenia": "AM",
    "Aruba": "AW",
    "Australia": "AU",
    "Australia, Egypt": ["AU", "EG"],
    "Australia, Fiji": ["AU", "FJ"],
    "Australia, Lebanon": ["AU", "LB"],
    "Austria": "AT",
    "Azerbaijan": "AZ",
    "Azerbaijan, Rep. of": "AZ",
    "BA, HR": ["BA", "HR"],
    "Bahamas, The": "BS",
    "Bahrain": "BH",
    "Bangladesh": "BD",
    "Bangladeshi": "BD",
    "Barbados": "BB",
    "Belarus": "BY",
    "Belgium": "BE",
    "Belgian": "BE",
    "Belgian Those travelling or staying abroad are urged to be vigilant.": "BE",
    "Belize": "BZ",
    "Benin": "BJ",
    "Bermuda": "BM",
    "Bhutan - NDE": "BT",
    "Bolivia": "BO",
    "Bolivia-Non ADB Member Country": "BO",
    "Bosnia and Herzegovina": "BA",
    "Botswana": "BW",
    "Brazil": "BR",
    "Brazil-Non ADB Member Country": "BR",
    "British": "GB",
    "Brunei": "BN",
    "Bulgaria": "BG",
    "Burkina Faso": "BF",
    "Burma": "MM",
    "Myanmar": "MM",
    "Myanmar (Burma)": "MM",
    "Burundi": "BI",
    "Cabo Verde": "CV",
    "Cambodia": "KH",
    "Cameroon": "CM",
    "Canada": "CA",
    "Central African Republic": "CF",
    "Central African Republic, Chad": ["CF", "TD"],
    "Central African Republic, Sudan": ["CF", "SD"],
    "Chad": "TD",
    "Chad\n, Central African Republic": ["TD", "CF"],
    "Chile": "CL",
    "China": "CN",
    "Chinese": "CN",
    "People's Republic of China": "CN",
    "Colombia": "CO",
    "Colombia-Non ADB Member Country": "CO",
    "Comoros": "KM",
    "Congo Democratic Republic Of": "CD",
    "Congo, Democratic Republic of the": "CD",
    "Democratic Republic of the Congo": "CD",
    "Congo, Republic of the": "CG",
    "Cook Islands": "CK",
    "Costa Rica": "CR",
    "Costa Rica-Non ADB Member Country": "CR",
    "Cote d'Ivoire": "CI",
    "Croatia": "HR",
    "Cuba": "CU",
    "Cyprus": "CY",
    "Czechia": "CZ",
    "DE, LY": ["DE", "LY"],
    "Democratic People's Republic of Korea": "KP",
    "Democratic People's Republic of Korea, Democratic People’s Republic of Korea": "KP",
    "Democratic Peoples Republic of Korea": "KP",
    "Korea, North - NDE": "KP",
    "Korea, South": "KR",
    "Denmark": "DK",
    "Djibouti": "DJ",
    "Dominica": "DM",
    "Dominican Republic": "DO",
    "Ecuador": "EC",
    "Ecuador-Non ADB Member Country": "EC",
    "Egypt": "EG",
    "Egyptian": "EG",
    "El Salvador": "SV",
    "El Salvador-Non ADB Member Country": "SV",
    "Equatorial Guinea": "GQ",
    "Eritrea": "ER",
    "Estonia": "EE",
    "Eswatini": "SZ",
    "Ethiopia": "ET",
    "Possibly Ethiopian": "ET",
    "Fiji": "FJ",
    "Philippines": "PH",
    "Filipino": "PH",
    "Philipino": "PH",
    "Philippines": "PH",
    "Finland": "FI",
    "France": "FR",
    "French": "FR",
    "France, Tunisia": ["FR", "TN"],
    "Gabon": "GA",
    "Gambia, The": "GM",
    "Georgia": "GE",
    "Georgian": "GE",
    "German": "DE",
    "Germany": "DE",
    "Germany, Algeria": ["DE", "DZ"],
    "Germany, Morocco": ["DE", "MA"],
    "Ghana": "GH",
    "Greece": "GR",
    "Grenada": "GD",
    "Guatemala": "GT",
    "Guatemala-Non ADB Member Country": "GT",
    "Guinea": "GN",
    "Guinea-Bisau, Guinea-Bissau": "GW",
    "Guinea-Bissau": "GW",
    "Guyana": "GY",
    "Guyana-Non ADB Member Country": "GY",
    "Haiti": "HT",
    "Haitian": "HT",
    "Holy See (Vatican City)": "VA",
    "Honduras": "HN",
    "Honduras - Non ADB Member Country": "HN",
    "Hong Kong, China": "HK",
    "Hungary": "HU",
    "IRAQ": "IQ",
    "Iraq": "IQ",
    "Iraqi": "IQ",
    "Iceland": "IS",
    "India": "IN",
    "Indian": "IN",
    "Indonesia": "ID",
    "Indonesia (as at Dec. 2003)": "ID",
    "Indonesian": "ID",
    "Indonesian (as at December 2003)": "ID",
    "Iran": "IR",
    "Irani": "IR",
    "Iraní": "IR",
    "Iran (Islamic Republic of)": "IR",
    "Iran - NDE": "IR",
    "Iran, United States": ["IR", "US"],
    "Isfahan, Iran": "IR",
    "Ireland": "IE",
    "Israel": "IL",
    "Italy": "IT",
    "Jamaica": "JM",
    "Japan": "JP",
    "Jordan": "JO",
    "Jordanian": "JO",
    "Jordan, United States of America": ["JO", "US"],
    "Kazakhstan": "KZ",
    "Kazakhstan, Ukraine": ["KZ", "UA"],
    "Kenya": "KE",
    "Kenya, Somalia": ["KE", "SO"],
    "Kenya-Non ADB Member Country": "KE",
    "Kiribati": "KI",
    "Kosovo": "XK",
    "Kuwait": "KW",
    "Kuwaiti": "KW",
    "Kuwaiti citizenship withdrawn": "KW",
    "Kuwaiti citizenship withdrawn in 2002": "KW",
    "Kyrgyz Republic": "KG",
    "Kyrgyzstan": "KG",
    "Lao People's Democratic Republic": "LA",
    "Laos": "LA",
    "Latvia": "LV",
    "Latvian": "LV",
    "Lebanon": "LB",
    "Lesotho": "LS",
    "Liberia": "LR",
    "Liberia-Non ADB Member Country": "LR",
    "Libya": "LY",
    "Libyan": "LY",
    "Liechtenstein": "LI",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Madagascar": "MG",
    "Malawi": "MW",
    "Malaysia": "MY",
    "Malaysia, Indonesia": ["MY", "ID"],
    "Maldives": "MV",
    "Mali": "ML",
    "Malian": "ML",
    "Mali, Mauritius": ["ML", "MU"],
    "Malta": "MT",
    "Marshall Islands": "MH",
    "Mauritania": "MR",
    "Mauritanian": "MR",
    "Mautitanian": "MR",
    "Mauritius": "MU",
    "Mexico": "MX",
    "Micronesia, Federated States of": "FM",
    "Moldova": "MD",
    "Moldova, Russia, Ukraine": ["MD", "RU", "UA"],
    "Monaco": "MC",
    "Mongolia": "MN",
    "Montenegro": "ME",
    "Morocco": "MA",
    "Moroccan": "MA",
    "Mozambique": "MZ",
    "Namibia": "NA",
    "Nauru": "NR",
    "Nepal": "NP",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Nicaragua": "NI",
    "Niger": "NE",
    "Nigeria": "NG",
    "Nigerian": "NG",
    "Nigeria-Non ADB Member Country": "NG",
    "Niue": "NU",
    "North Macedonia": "MK",
    "Norway": "NO",
    "Norwegian": "NO",
    "Oman": "OM",
    "Omani": "OM",
    "Pakistan, Saudi Arabia": ["PK", "SA"],
    "Palau": "PW",
    "Palestina.": "PS",
    "Palestine": "PS",
    "State of Palestine": "PS",
    "Palestinian": "PS",
    "Panama": "PA",
    "Papua New Guinea": "PG",
    "Paraguay": "PY",
    "Peru": "PE",
    "Peru-Non ADB Member Country": "PE",
    "Poland": "PL",
    "Portugal": "PT",
    "Qatar": "QA",
    "Qatari": "QA",
    "Republic of Tajikistan": "TJ",
    "Tajikistan": "TJ",
    "Tajikistan, Rep. of": "TJ",
    "Romania": "RO",
    "Russia": "RU",
    "Russian": "RU",
    "Russian Federation": "RU",
    "Russia, Bulgaria": ["RU", "BG"],
    "Russia, Estonia": ["RU", "EE"],
    "Russia, Georgia": ["RU", "GE"],
    "Russia, German": ["RU", "DE"],
    "Russia, Swiss": ["RU", "CH"],
    "Russia, Ukraine": ["RU", "UA"],
    "Ukraine, Russia": ["UA", "RU"],
    "Russia, Uzbekistan": ["RU", "UZ"],
    "Rwanda": "RW",
    "Saint Kitts and Nevis": "KN",
    "Saint Lucia": "LC",
    "Saint Vincent and the Grenadines": "VC",
    "Samoa": "WS",
    "San Marino": "SM",
    "Sao Tome and Principe": "ST",
    "Saudi": "SA",
    "Saudi Arabia": "SA",
    "Saudi Arabian": "SA",
    "Senegal": "SN",
    "Senegal-Non ADB Member Country": "SN",
    "Senegalese": "SN",
    "Serbia": "RS",
    "Serbian": "RS",
    "Seychelles": "SC",
    "Sierra Leone": "SL",
    "Singapore": "SG",
    "Slovak Republic, Azerbaijan": ["SK", "AZ"],
    "Slovakia": "SK",
    "Slovenia": "SI",
    "Socialist Republic of Viet Nam": "VN",
    "Vietnam": "VN",
    "Solomon Islands": "SB",
    "Somali": "SO",
    "Somalia": "SO",
    "Somalia, Kenya": ["SO", "KE"],
    "South Africa": "ZA",
    "South Sudan": "SS",
    "South Sudan, Central African Republic": ["SS", "CF"],
    "South Sudan, Uganda": ["SS", "UG"],
    "Spain": "ES",
    "Sri Lanka": "LK",
    "Sudan": "SD",
    "Sudanese by birth": "SD",
    "a) Sudan b) South Sudan": ["SD", "SS"],
    "Suriname": "SR",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Syria": "SY",
    "Syrian Arab Republic": "SY",
    "Syria, France": ["SY", "FR"],
    "Syria, State of Palestine": ["SY", "PS"],
    "Taiwan - NDE": "TW",
    "Tanzania": "TZ",
    "United Republic of Tanzania": "TZ",
    "Tanzania-Non ADB Member Country": "TZ",
    "Thailand": "TH",
    "Timor-Leste": "TL",
    "Togo": "TG",
    "Tonga": "TO",
    "Trinidad and Tobago": "TT",
    "Trinidad and Tobago, United States of America": ["TT", "US"],
    "Tunisia": "TN",
    "Tunisian": "TN",
    "Turkey": "TR",
    "Turkish": "TR",
    "Türkiye": "TR",
    "Türkiye, Jordan": ["TR", "JO"],
    "Turkmenistan": "TM",
    "Tuvalu": "TV",
    "U.S.A": "US",
    "U.S.A.": "US",
    "United States": "US",
    "United States of America": "US",
    "United States of America, Yemen": ["US", "YE"],
    "United States. Also believed to hold Syrian nationality": ["US", "SY"],
    "Uganda": "UG",
    "Ukraine": "UA",
    "Ukraine, Uzbekistan": ["UA", "UZ"],
    "United Arab Emirates": "AE",
    "United Kingdom": "GB",
    "United Kingdom of Great Britain": "GB",
    "United Kingdom of Great Britain and Northern Ireland": "GB",
    "Uruguay": "UY",
    "Uzbek": "UZ",
    "Uzbekistan": "UZ",
    "Uzbekistan, Afghanistan": ["UZ", "AF"],
    "Uzbekistan, Rep. of": "UZ",
    "Vanuatu": "VU",
    "Venezuela": "VE",
    "Yemen": "YE",
    "Yemeni": "YE",
    "Zambia": "ZM",
    "Zimbabwe": "ZW",

    # --- Extra demonyms / odd-but-country (keep as country code) ---
    "LIBANÉS": "LB",
    "Libanesa": "LB",
    "Iraní": "IR",

    # --- Non-country / noise -> None ---
    "Extremist Settler Violence": None,
    "Hamas Terrorist Attacks": None,
    "Justice for Victims of Corrupt Foreign Officials Regulations (JVCFOR)": None,

    # --- Ukrainian names ---
    "Єгипет": "EG",
    "Ємен": "YE",
    "Ізраїль": "IL",
    "Індія": "IN",
    "Ірландія": "IE",
    "Ісламська Республіка Іран": "IR",
    "Іспанія": "ES",
    "Італія": "IT",
    "Австрія": "AT",
    "Азербайджан": "AZ",
    "Бангладеш": "BD",
    "Бельгія": "BE",
    "Болгарія": "BG",
    "Боснія і Герцеговина": "BA",
    "Бразилія": "BR",
    "Велика Британія": "GB",
    "Венесуела": "VE",
    "Вірменія": "AM",
    "Гвінея-Бісау": "GW",
    "Гренада": "GD",
    "Греція": "GR",
    "Грузія": "GE",
    "Естонія": "EE",
    "Казахстан": "KZ",
    "Канада": "CA",
    "Киргизстан": "KG",
    "Китай": "CN",
    "Кот-Д'Івуар": "CI",
    "Кіпр": "CY",
    "Латвія": "LV",
    "Литовська Республіка": "LT",
    "Ліхтенштейн": "LI",
    "М'янма": "MM",
    "Мальта": "MT",
    "Малі": "ML",
    "Молдова": "MD",
    "Монако": "MC",
    "Нова Зеландія": "NZ",
    "Нідерланди": "NL",
    "Німеччина": "DE",
    "Об'єднані Арабські Емірати": "AE",
    "Оман": "OM",
    "Пакистан": "PK",
    "Палестина": "PS",
    "Польща": "PL",
    "Південна Африка": "ZA",
    "Південна Корея": "KR",
    "Південний Судан": "SS",
    "Північна Корея": "KP",
    "Республіка Білорусь": "BY",
    "Російська Федерація": "RU",
    "Румунія": "RO",
    "США": "US",
    "Сент-Кітс і Невіс": "KN",
    "Сербія": "RS",
    "Сингапур": "SG",
    "Сирія": "SY",
    "Словаччина": "SK",
    "Судан": "SD",
    "Того": "TG",
    "Туреччина": "TR",
    "Уганда": "UG",
    "Угорщина": "HU",
    "Узбекистан": "UZ",
    "Україна": "UA",
    "Уругвай": "UY",
    "Франція": "FR",
    "Фінляндія": "FI",
    "Хорватія": "HR",
    "Чехія": "CZ",
    "Чорногорія": "ME",
    "Швейцарія": "CH",
    "Швеція": "SE",

    # --- Hebrew names / mixed strings ---
    "אוזבקיסטן / אפגניסטן": ["UZ", "AF"],
    "אזרחות פלסטינית": "PS",
    "הרשות הפלסטינית": "PS",
    "אלג'יר או רש\"פ": ["DZ", "PS"],
    "אינדונזיה": "ID",
    "אירן": "IR",
    "אלג'יר": "DZ",
    "אלג'יריה": "DZ",
    "אפגניסטן": "AF",
    "אתיופיה": "ET",
    "בריטניה": "GB",
    "גרמניה": "DE",
    "דרום אפריקה": "ZA",
    "טנזניה": "TZ",
    "טרינידד וטובאגו": "TT",
    "טרינידד וטובגו": "TT",
    "ירדן": "JO",
    "ירדן, כווית": ["JO", "KW"],
    "לבנון": "LB",
    "מאלזיה": "MY",
    "מאלי": "ML",
    "מרוקו": "MA",
    "סוריה": "SY",
    "סין": "CN",
    "סעודיה": "SA",
    "ערב הסעודית": "SA",
    "עיראק": "IQ",
    "עירק": "IQ",
    "עירק/סוריה": ["IQ", "SY"],
    "פיליפים / סעודיה": ["PH", "SA"],
    "פיליפינים": "PH",
    "פקיסטן": "PK",
    "צרפת": "FR",
    "רוסיה": "RU",
    "תוניסיה": "TN",
    "תורכיה": "TR",
    "תימן": "YE",
}





_NOISE_TAIL_RE = re.compile(
    r"""
    \s*(
        -\s*NDE
      | -\s*Non\s*ADB\s*Member\s*Country
      | \(\s*as\s*at\s*Dec\.?\s*\d{4}\s*\)
      | \(\s*as\s*at\s*December\s*\d{4}\s*\)
      | \(\s*as\s*at\s*\w+\s*\d{4}\s*\)
    )\s*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

def _as_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v is not None]
    return [str(x)]

def _clean_token(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").strip()
    if not s:
        return ""
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip()
    s = _NOISE_TAIL_RE.sub("", s).strip()
    s = re.sub(r"\.+$", "", s).strip()
    return s

def normalize_nationalities(value: Any) -> list[str]:
    """
    Convert raw nationality/citizenship to ISO-2 list using COUNTRY_TO_ISO2.
    - Keeps existing ISO-2 codes (len==2)
    - Splits combined strings (; , / | & and)
    - Maps tokens via COUNTRY_TO_ISO2
    """
    raw_items = _as_list(value)
    raw_items = [s for s in raw_items if s and s.strip() and s.strip().lower() not in {"nan", "none", "null"}]
    if not raw_items:
        return []

    out: list[str] = []
    seen: set[str] = set()

    def push(code: str) -> None:
        code = (code or "").strip().upper()
        if code and code not in seen:
            seen.add(code)
            out.append(code)

    for raw in raw_items:
        raw = raw.strip()
        if not raw:
            continue

        # If it's already exactly ISO2, keep it
        if re.fullmatch(r"[A-Za-z]{2}", raw):
            push(raw)
            continue

        # Split combined strings into tokens
        tokens = [t for t in _SPLIT_RE.split(raw) if t and t.strip()]
        for tok in tokens:
            tok = _clean_token(tok)
            if not tok:
                continue

            # If token is ISO2 already
            if re.fullmatch(r"[A-Za-z]{2}", tok):
                push(tok)
                continue

            mapped = COUNTRY_TO_ISO2.get(tok)

            if isinstance(mapped, str):
                push(mapped)
            elif isinstance(mapped, list):
                for c in mapped:
                    if isinstance(c, str):
                        push(c)
            else:
                # fallback attempt: remove Rep./Republic wording
                tok2 = re.sub(r"\bRep\.?\s*of\b", "", tok, flags=re.IGNORECASE).strip()
                tok2 = re.sub(r"\bRepublic\s+of\b", "", tok2, flags=re.IGNORECASE).strip()
                tok2 = _clean_token(tok2)

                mapped2 = COUNTRY_TO_ISO2.get(tok2)
                if isinstance(mapped2, str):
                    push(mapped2)
                elif isinstance(mapped2, list):
                    for c in mapped2:
                        if isinstance(c, str):
                            push(c)

    return out

def normalize_country_fields_in_profile(data: dict) -> None:
    """
    Mutates profile['data'] in-place:
    - Ensures nationalitiesIsoCodes is ISO-2 list
    - Ensures nationality is also ISO-2 list (optional but keeps consistency)
    """
    # Collect from both fields because some sources put names in either one
    raw = []
    raw.extend(_as_list(data.get("nationalitiesIsoCodes")))
    raw.extend(_as_list(data.get("nationality")))

    iso = normalize_nationalities(raw)

    data["nationalitiesIsoCodes"] = iso
    data["nationality"] = iso  # keep consistent (optional)
