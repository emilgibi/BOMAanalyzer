"""
BOM Analyzer Web Edition v1.0.0
Faithful Streamlit conversion of Tyler Allen's BOM_Analyzer desktop app.
Component Revision Delta (or Difference) Verification - CRDV
- Original risk scoring, strategy engine, and buy-up logic preserved exactly
- OpenAI replaced with Groq (free tier)
- Tkinter GUI replaced with Streamlit
- DigiKey OAuth replaced with Mouser + Nexar (no localhost callback needed)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
import re
import io
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BOM Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .title-bar { font-size:2rem; font-weight:700; color:#0078d4; margin-bottom:0; }
  .subtitle  { font-size:.95rem; color:#555; margin-bottom:1.5rem; }
  .section-head { font-size:1rem; font-weight:700; color:#0078d4; border-bottom:2px solid #0078d4;
                  padding-bottom:4px; margin-bottom:.8rem; }
  .risk-high   { background:#fee2e2; border-left:4px solid #d13438; padding:5px 10px; border-radius:4px; }
  .risk-mod    { background:#fef3c7; border-left:4px solid #ca5010; padding:5px 10px; border-radius:4px; }
  .risk-low    { background:#dcfce7; border-left:4px solid #107c10; padding:5px 10px; border-radius:4px; }
  .kpi-box { background:#f0f4fa; border-radius:8px; padding:1rem; border-left:4px solid #0078d4; }
</style>
""", unsafe_allow_html=True)

# ── Constants (exact from source) ─────────────────────────────────────────────
RISK_WEIGHTS     = {'Sourcing': 0.30, 'Stock': 0.15, 'LeadTime': 0.15, 'Lifecycle': 0.30, 'Geographic': 0.10}
GEO_RISK_TIERS   = {
    "China":7,"Russia":9,"Taiwan":5,"Malaysia":4,"Vietnam":4,"India":5,"Philippines":4,
    "Thailand":4,"South Korea":3,"USA":1,"United States":1,"Mexico":2,"Canada":1,"Japan":1,
    "Germany":1,"France":1,"UK":1,"Ireland":1,"Switzerland":1,"EU":1,
    "Unknown":4,"N/A":4,"_DEFAULT_":4,
}
RISK_CATEGORIES  = {'high':(6.6,10.0), 'moderate':(3.6,6.5), 'low':(0.0,3.5)}
API_TIMEOUT      = 20
MAX_WORKERS      = 6

# Country name → ISO2 helpers for COO matching
COUNTRY_ISO = {
    "CN":"China","TW":"Taiwan","US":"United States","MX":"Mexico","DE":"Germany",
    "JP":"Japan","KR":"South Korea","MY":"Malaysia","VN":"Vietnam","IN":"India",
    "PH":"Philippines","TH":"Thailand","CA":"Canada","FR":"France","GB":"UK",
    "IE":"Ireland","CH":"Switzerland","RU":"Russia",
}

# ── Utility Functions (ported from source) ─────────────────────────────────────

def safe_float(value, default=np.nan):
    if value is None or isinstance(value, bool): return default
    if isinstance(value, (int,float)):
        return float(value) if not np.isinf(value) else default
    try:
        s = str(value).strip().replace('$','').replace(',','').replace('%','').lower()
        if not s or s in ['n/a','none','inf','-inf','na','nan','']: return default
        return float(s)
    except: return default


def convert_lead_time_to_days(val):
    """Exact port of source convert_lead_time_to_days."""
    if val is None or (isinstance(val, float) and np.isnan(val)): return np.nan
    if isinstance(val, (int, float)):
        return int(round(val)) if not np.isinf(val) else np.nan
    s = str(val).lower().strip()
    if s in ['n/a','unknown','','na','none']: return np.nan
    if s == 'stock': return 0
    try:
        m = re.search(r'(\d+(\.\d+)?)', s)
        if not m: return np.nan
        num = float(m.group(1))
        if 'week' in s: return int(round(num * 7))
        return int(round(num))
    except: return np.nan


def get_optimal_cost(qty_needed, pricing_breaks, min_order_qty=0, buy_up_threshold_pct=1.0):
    """
    Exact port of BOMAnalyzerApp.get_optimal_cost.
    Returns (unit_price, total_cost, actual_order_qty, notes)
    """
    notes = ""
    if not isinstance(qty_needed, (int,float)) or qty_needed <= 0:
        return np.nan, np.nan, qty_needed, "Invalid Qty Needed"
    if not isinstance(pricing_breaks, list):
        return np.nan, np.nan, qty_needed, "Invalid Pricing Data"

    try:
        valid_breaks = [
            {'qty': int(pb['qty']), 'price': safe_float(pb['price'])}
            for pb in pricing_breaks
            if isinstance(pb, dict) and 'qty' in pb and 'price' in pb
            and int(pb['qty']) > 0 and pd.notna(safe_float(pb['price']))
            and safe_float(pb['price']) >= 0
        ]
        if not valid_breaks:
            return np.nan, np.nan, qty_needed, "No Valid Price Breaks"
        pricing_breaks = sorted(valid_breaks, key=lambda x: x['qty'])
        min_order_qty  = max(1, int(safe_float(min_order_qty, default=1)))
    except Exception as e:
        return np.nan, np.nan, qty_needed, f"Pricing Data Error: {e}"

    base_order_qty = max(int(qty_needed), min_order_qty)

    base_unit_price = np.nan
    applicable_break = None
    for pb in pricing_breaks:
        if base_order_qty >= pb['qty']:
            applicable_break = pb
        else:
            break
    if applicable_break:
        base_unit_price = applicable_break['price']
    elif pricing_breaks:
        applicable_break = pricing_breaks[0]
        base_unit_price  = applicable_break['price']
        base_order_qty   = max(base_order_qty, applicable_break['qty'])
        notes += f"MOQ adjusted to first break ({base_order_qty}). "
    else:
        return np.nan, np.nan, qty_needed, "Cannot Determine Base Price"

    best_total_cost  = base_unit_price * base_order_qty
    best_unit_price  = base_unit_price
    actual_order_qty = base_order_qty

    for pb in pricing_breaks:
        break_qty   = pb['qty']
        break_price = pb['price']
        if break_qty >= base_order_qty:
            total_cost_at_break = break_qty * break_price
            if total_cost_at_break < best_total_cost * (1.0 - (buy_up_threshold_pct / 100.0)):
                best_total_cost  = total_cost_at_break
                best_unit_price  = break_price
                actual_order_qty = break_qty
                notes = f"Price break @ {break_qty} lower total cost. "
            elif (actual_order_qty < break_qty and
                  total_cost_at_break <= best_total_cost * (1.0 + (buy_up_threshold_pct / 100.0))):
                best_total_cost  = total_cost_at_break
                best_unit_price  = break_price
                actual_order_qty = break_qty
                notes = f"Bought up to {break_qty} for similar total cost. "

    return best_unit_price, best_total_cost, actual_order_qty, notes.strip()


def calculate_risk_score(sourcing_count, stock_available, qty_needed,
                         lead_time_days, lifecycle_notes, coo):
    """
    Exact port of source risk factor logic from analyze_single_part.
    Returns (overall_score 0-10, risk_factors dict)
    """
    risk_factors = {}

    # Sourcing risk  (0, 4, 7, 10)
    if sourcing_count == 0:   risk_factors['Sourcing'] = 10
    elif sourcing_count == 1: risk_factors['Sourcing'] = 7
    elif sourcing_count == 2: risk_factors['Sourcing'] = 4
    else:                     risk_factors['Sourcing'] = 0

    # Stock risk
    has_stock_gap = (stock_available < qty_needed)
    if has_stock_gap:                                    risk_factors['Stock'] = 8
    elif stock_available < 1.5 * qty_needed:             risk_factors['Stock'] = 4
    else:                                                risk_factors['Stock'] = 0

    # Lead time risk (based on fastest available)
    lt = lead_time_days
    if pd.isna(lt) or lt == np.inf:       risk_factors['LeadTime'] = 9
    elif lt == 0:                          risk_factors['LeadTime'] = 0
    elif lt > 90:                          risk_factors['LeadTime'] = 7
    elif lt > 45:                          risk_factors['LeadTime'] = 4
    else:                                  risk_factors['LeadTime'] = 1

    # Lifecycle risk
    lc = str(lifecycle_notes).upper()
    if "EOL" in lc or "DISC" in lc:       risk_factors['Lifecycle'] = 10
    else:                                  risk_factors['Lifecycle'] = 0

    # Geographic risk
    # Try to match country name from COO string
    coo_str = str(coo).strip()
    geo_score = GEO_RISK_TIERS.get("_DEFAULT_", 4)
    for country, score in GEO_RISK_TIERS.items():
        if country.lower() in coo_str.lower():
            geo_score = score
            break
    risk_factors['Geographic'] = geo_score

    overall = sum(risk_factors[f] * RISK_WEIGHTS[f] for f in RISK_WEIGHTS)
    overall = round(max(0.0, min(10.0, overall)), 1)
    return overall, risk_factors


def get_tariff_rate(coo, custom_tariffs):
    """Map COO to tariff rate using custom tariff table or defaults."""
    coo_str = str(coo).strip().lower()
    for country, rate in custom_tariffs.items():
        if country.lower() in coo_str:
            return rate
    # Default tariff
    if "china" in coo_str or "cn" == coo_str: return 0.25
    if "taiwan" in coo_str or "tw" == coo_str: return 0.0
    return 0.035  # WTO baseline


# ── Part Number Cleaner ───────────────────────────────────────────────────────

def clean_part_number(pn):
    """
    Clean and normalize part numbers before sending to supplier APIs.
    Handles all patterns found in real-world PCB BOM exports:
    - Leading apostrophes from Excel  ('9001-12-01 → 9001-12-01)
    - PBFREE / PB FREE suffix         (2N3906 PBFREE → 2N3906)
    - Trailing/leading whitespace     (C3216X7R2E104K160AA  → C3216X7R2E104K160AA)
    - Embedded newlines from CSV      (cleaned at CSV-read stage)
    - ADI #PBF suffix                 (LTC2057HVHS8#PBF → KEEP as-is, Mouser recognizes it)
    Returns (cleaned_pn, original_pn, list_of_changes)
    """
    original = pn
    changes  = []
    p        = str(pn).strip()

    # Remove leading apostrophe/quote (Excel CSV artifact — Excel prepends ' to
    # prevent numeric interpretation of part numbers like 9001-12-01)
    if p.startswith(("'", '"', "`")):
        p = p.lstrip("'\"`")
        changes.append("removed leading apostrophe (Excel artifact)")

    # Remove PBFREE / PB-FREE / PB FREE suffix (RoHS marker, not part of MPN)
    # Important: Do NOT strip #PBF from ADI/LTC parts — it IS part of their MPN
    pbfree_variants = [" PBFREE", "-PBFREE", "_PBFREE", " PB-FREE", "-PB-FREE", " PB FREE"]
    p_upper = p.upper()
    for suffix in pbfree_variants:
        if p_upper.endswith(suffix):
            p = p[:len(p) - len(suffix)].strip(" -_")
            changes.append(f"removed '{suffix.strip()}' suffix")
            break

    # Remove common distributor/packaging suffixes that aren't part of the base MPN
    # Exception: #PBF is kept because ADI/LTC use it as part of their official MPN
    dist_suffixes = ["-ND", "-1-ND", "-2-ND", "-TR", "-T&R", "-TRC",
                     "-CT", "-CUT", "-REEL", "/T", "-T"]
    p_upper = p.upper()
    for suffix in dist_suffixes:
        if p_upper.endswith(suffix.upper()):
            p = p[:len(p) - len(suffix)].strip(" -_")
            changes.append(f"removed distributor suffix '{suffix}'")
            break

    # Final whitespace cleanup
    p_clean = p.strip()
    if p_clean != original.strip() and not changes:
        changes.append("stripped whitespace")

    return p_clean, original, changes


# ── API Functions ─────────────────────────────────────────────────────────────

def search_mouser(part_number, api_key):
    """Fetch from Mouser API. Returns standardized result dict or None."""

    print(f"[MOUSER] search_mouser called for PN: {part_number}")

    if not api_key:
        print("[MOUSER] ❌ No API key provided")
        return None

    url = "https://api.mouser.com/api/v1/search/partnumber"
    params = {"apiKey": api_key}
    payload = {
        "SearchByPartRequest": {
            "mouserPartNumber": part_number,
            "partSearchOptions": "string"
        }
    }

    try:
        print("[MOUSER] 📡 Sending API request")
        r = requests.post(
            url,
            params=params,
            json=payload,
            timeout=API_TIMEOUT
        )

        print(f"[MOUSER] HTTP status: {r.status_code}")
        r.raise_for_status()

        data = r.json()
        parts = data.get("SearchResults", {}).get("Parts", [])

        print(f"[MOUSER] Parts returned: {len(parts)}")

        if not parts:
            print("[MOUSER] ❌ No parts found in Mouser catalog")
            return None

        p = parts[0]
        print(
            f"[MOUSER] ✅ Using first result: "
            f"MPN={p.get('ManufacturerPartNumber')}, "
            f"MouserPN={p.get('MouserPartNumber')}"
        )

        # ───────────── Price breaks ─────────────
        price_breaks = []
        for pb in p.get("PriceBreaks", []):
            try:
                qty = int(pb.get("Quantity", 0))
                raw_price = str(pb.get("Price", "") or "").strip().replace("$", "").replace(",", "").replace(" ", "")
                price = safe_float(raw_price if raw_price else "0")
                if qty > 0 and pd.notna(price) and price > 0:
                    price_breaks.append({"qty": qty, "price": price})
            except Exception as e:
                print(f"[MOUSER] ⚠️ PriceBreak parse error: {e}")

        # Fallback: if no price breaks, try top-level UnitPrice field
        if not price_breaks:
            unit_price_raw = str(p.get("UnitPrice", "") or "").strip().replace("$", "").replace(",", "").replace(" ", "")
            unit_price = safe_float(unit_price_raw)
            moq = int(safe_float(p.get("Min", "1"), default=1))
            if pd.notna(unit_price) and unit_price > 0:
                price_breaks.append({"qty": moq, "price": unit_price})
                print(f"[MOUSER] ⚠️ Used UnitPrice fallback: qty={moq}, price={unit_price}")

        print(f"[MOUSER] Pricing tiers found: {price_breaks}")

        # ───────────── Lead time ─────────────
        raw_lt = p.get("LeadTime", "")
        lt_days = convert_lead_time_to_days(raw_lt)
        print(f"[MOUSER] Lead time raw='{raw_lt}' → {lt_days} days")

        # ───────────── Lifecycle ─────────────
        eol = (p.get("LifecycleStatus") or "").upper()
        is_eol = any(
            x in eol
            for x in ["OBSOLETE", "EOL", "DISCONTINUED", "NOT RECOMMENDED"]
        )
        is_disc = "DISCONTINUED" in eol

        print(
            f"[MOUSER] LifecycleStatus='{eol}', "
            f"EOL={is_eol}, Discontinued={is_disc}"
        )

        # ───────────── Stock ─────────────
        stock = int(safe_float(
            p.get("AvailabilityInStock", 0),
            default=0
        ))

        print(f"[MOUSER] Stock available: {stock}")

        print("[MOUSER] ✅ Returning Mouser result")

        return {
            "Source":                 "Mouser",
            "SourcePartNumber":       p.get("MouserPartNumber", "N/A"),
            "ManufacturerPartNumber": p.get("ManufacturerPartNumber", part_number),
            "Manufacturer":           p.get("Manufacturer", "N/A"),
            "Description":            p.get("Description", ""),
            "Stock":                  stock,
            "LeadTimeDays":           lt_days,
            "MinOrderQty":            int(safe_float(p.get("Min", "1"), default=1)),
            "Pricing":                price_breaks,
            "CountryOfOrigin":        p.get("CountryOfOrigin", "Unknown"),
            "NormallyStocking":       True,
            "Discontinued":           is_disc,
            "EndOfLife":              is_eol,
            "DatasheetUrl":           p.get("DataSheetUrl", ""),
        }

    except Exception as e:
        print(f"[MOUSER] ❌ API error: {e}")
        return None

def search_nexar(part_number, client_id, client_secret, _token_cache):
    """Fetch from Nexar (Octopart) GraphQL API. Returns standardized result dict or None."""

    print(f"[NEXAR] search_nexar called for PN: {part_number}")

    if not client_id or not client_secret:
        print("[NEXAR] ❌ Missing client_id or client_secret")
        return None

    # ───────────────── Token handling ─────────────────
    now = time.time()
    if _token_cache.get("expires_at", 0) > now + 60:
        token = _token_cache["access_token"]
        print("[NEXAR] ✅ Using cached OAuth token")
    else:
        try:
            print("[NEXAR] 🔑 Requesting new OAuth token")
            tr = requests.post(
                "https://identity.nexar.com/connect/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": "supply.domain",
                },
                timeout=API_TIMEOUT,
            )
            print(f"[NEXAR] OAuth HTTP status: {tr.status_code}")
            tr.raise_for_status()
            td = tr.json()

            token = td.get("access_token")
            if not token:
                print(f"[NEXAR] ❌ OAuth response missing access_token: {td}")
                return None

            _token_cache["access_token"] = token
            _token_cache["expires_at"] = now + td.get("expires_in", 3600) - 60
            print("[NEXAR] ✅ OAuth token acquired and cached")
        except Exception as e:
            print(f"[NEXAR] ❌ OAuth token error: {e}")
            return None

    # ───────────────── GraphQL query ─────────────────
    query = """
    query Search($q: String!) {
      supSearchMpn(q: $q, limit: 1) {
        results {
          part {
            mpn
            shortDescription
            manufacturer { name }
            bestDatasheet { url }
            sellers(includeBrokers: false) {
              company { name }
              offers {
                sku
                inventoryLevel
                moq
                factoryLeadDays
                packaging
                prices { quantity price currency }
              }
            }
          }
        }
      }
    }
    """

    try:
        print("[NEXAR] 📡 Sending GraphQL request")
        r = requests.post(
            "https://api.nexar.com/graphql",
            json={"query": query, "variables": {"q": part_number}},
            headers={"Authorization": f"Bearer {token}"},
            timeout=API_TIMEOUT,
        )
        print(f"[NEXAR] GraphQL HTTP status: {r.status_code}")
        r.raise_for_status()

        response_json = r.json()
        hits = (
            response_json
            .get("data", {})
            .get("supSearchMpn", {})
            .get("results", [])
        )

        print(f"[NEXAR] Hits returned: {len(hits)}")

        # Surface any GraphQL errors for easier debugging
        if response_json.get("errors"):
            print(f"[NEXAR] ❌ GraphQL errors: {response_json['errors']}")
            return None

        if not hits:
            print("[NEXAR] ❌ No hits found for part")
            return None

        part_data = hits[0].get("part") if hits[0] else None
        if not part_data:
            print("[NEXAR] ❌ First hit has no part data")
            return None
        sellers = part_data.get("sellers", [])
        print(f"[NEXAR] Sellers found: {len(sellers)}")

        best_offer = None
        best_price = float("inf")
        best_seller = ""

        # ───────────────── Offer evaluation ─────────────────
        for seller in sellers:
            seller_name = seller.get("company", {}).get("name", "UNKNOWN")
            offers = seller.get("offers", [])
            print(f"[NEXAR]  Seller '{seller_name}' offers: {len(offers)}")

            for offer in offers:
                print(
                    f"[NEXAR]   Offer SKU={offer.get('sku')} "
                    f"Stock={offer.get('inventoryLevel')} "
                    f"Prices={len(offer.get('prices', []))}"
                )

                prices = offer.get("prices", [])

                if prices:
                    valid_prices = [
                        safe_float(p.get("price"))
                        for p in prices
                        if pd.notna(safe_float(p.get("price")))
                    ]
                    print(f"[NEXAR]    Valid prices: {valid_prices}")

                    if valid_prices:
                        min_p = min(valid_prices)
                        if min_p < best_price:
                            best_price = min_p
                            best_offer = offer
                            best_seller = seller_name
                            print(f"[NEXAR] ✅ New best price: {min_p}")
                else:
                    if offer.get("inventoryLevel", 0) > 0 and best_offer is None:
                        best_offer = offer
                        best_seller = seller_name
                        print("[NEXAR] ⚠️ Stock-only offer selected (quote required)")

        if not best_offer:
            print("[NEXAR] ❌ No valid offers selected")
            return None

        # ───────────────── Pricing normalization ─────────────────
        pricing = sorted(
            [
                {"qty": int(p["quantity"]), "price": safe_float(p["price"])}
                for p in best_offer.get("prices", [])
                if p.get("currency") == "USD"
                and pd.notna(safe_float(p.get("price")))
            ],
            key=lambda x: x["qty"],
        )

        print(f"[NEXAR] Final pricing tiers (USD): {pricing}")

        lt_raw = best_offer.get("factoryLeadDays")
        lt_days = int(safe_float(lt_raw)) if pd.notna(safe_float(lt_raw)) else np.nan

        print(
            f"[NEXAR] ✅ Returning result: "
            f"Seller={best_seller}, Stock={best_offer.get('inventoryLevel')}, "
            f"LeadTime={lt_days}"
        )

        return {
            "Source": best_seller or "Octopart (Nexar)",
            "SourcePartNumber": best_offer.get("sku", "N/A"),
            "ManufacturerPartNumber": part_data.get("mpn", part_number),
            "Manufacturer": (part_data.get("manufacturer") or {}).get("name", "N/A"),
            "Description": part_data.get("shortDescription", ""),
            "Stock": int(safe_float(best_offer.get("inventoryLevel", 0), default=0)),
            "LeadTimeDays": lt_days,
            "MinOrderQty": int(safe_float(best_offer.get("moq", 1), default=1)),
            "Pricing": pricing,
            "CountryOfOrigin": "Unknown",
            "NormallyStocking": True,
            "Discontinued": False,
            "EndOfLife": False,
            "DatasheetUrl": (part_data.get("bestDatasheet") or {}).get("url", ""),
        }

    except Exception as e:
        print(f"[NEXAR] ❌ GraphQL processing error: {e}")
        return None

def get_part_data_parallel(part_number, mouser_key, nexar_id, nexar_secret, nexar_token_cache):
    """Fetch from all enabled suppliers in parallel. Returns dict of {supplier: result}."""
    results = {}
    tasks   = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="API") as ex:
        if mouser_key:
            tasks[ex.submit(search_mouser, part_number, mouser_key)] = "Mouser"
        if nexar_id and nexar_secret:
            tasks[ex.submit(search_nexar, part_number, nexar_id, nexar_secret, nexar_token_cache)] = "Nexar"

        for future in as_completed(tasks, timeout=30):
            name = tasks[future]
            try:
                result = future.result()
                if result and isinstance(result, dict):
                    results[name] = result
            except: pass
    return results


def analyze_single_part(bom_pn, bom_mfg, bom_qty_per_unit, config,
                        mouser_key, nexar_id, nexar_secret, nexar_token_cache):
    """
    Core single-part analysis. Faithful port of BOMAnalyzerApp.analyze_single_part.
    Returns dict with all fields for display and strategy engine.
    """
    total_units       = config.get("total_units", 100)
    buy_up_pct        = config.get("buy_up_threshold", 1.0)
    custom_tariffs    = config.get("custom_tariff_rates", {})
    total_qty_needed  = int(bom_qty_per_unit * total_units)

    # Fetch data
    supplier_data = get_part_data_parallel(bom_pn, mouser_key, nexar_id, nexar_secret, nexar_token_cache)

    # No data found case
    if not supplier_data:
        return {
            "PartNumber": bom_pn, "Manufacturer": bom_mfg or "N/A",
            "MfgPN": bom_pn, "QtyNeed": total_qty_needed,
            "Status": "Not Found", "Sources": "0", "StockAvail": 0,
            "COO": "Unknown", "RiskScore": 10.0, "TariffPct": "N/A",
            "BestCostPer": "N/A", "BestTotalCost": "N/A", "ActualBuyQty": "N/A",
            "BestCostLT": "N/A", "BestCostSrc": "N/A",
            "Description": "No supplier data — check API keys",
            "Notes": "No data", "_options": [], "_valid": False,
        }

    # Build options list
    all_options = []
    for src_name, sd in supplier_data.items():
        pricing = sd.get("Pricing", [])
        moq     = sd.get("MinOrderQty", 1)
        unit_p, total_c, act_qty, notes = get_optimal_cost(
            total_qty_needed, pricing, moq, buy_up_pct
        )
        stock   = sd.get("Stock", 0)
        lt_days = sd.get("LeadTimeDays", np.nan)
        if isinstance(lt_days, float) and np.isnan(lt_days):
            effective_lt = np.inf
        else:
            effective_lt = 0 if stock >= total_qty_needed else (lt_days if pd.notna(lt_days) else np.inf)

        all_options.append({
            "source":          sd.get("Source", src_name),
            "SourcePartNumber": sd.get("SourcePartNumber","N/A"),
            "ManufacturerPartNumber": sd.get("ManufacturerPartNumber", bom_pn),
            "Manufacturer":    sd.get("Manufacturer", bom_mfg or "N/A"),
            "Description":     sd.get("Description",""),
            "stock":           stock,
            "lead_time":       lt_days if pd.notna(lt_days) else np.inf,
            "effective_lead":  effective_lt,
            "unit_cost":       unit_p,
            "cost":            total_c,
            "actual_order_qty": act_qty,
            "notes":           notes,
            "coo":             sd.get("CountryOfOrigin","Unknown"),
            "eol":             sd.get("EndOfLife", False),
            "discontinued":    sd.get("Discontinued", False),
            "lifecycle":       "EOL" if sd.get("EndOfLife") else ("DISC" if sd.get("Discontinued") else "Active"),
            "DatasheetUrl":    sd.get("DatasheetUrl",""),
            "pricing":         pricing,
            "moq":             moq,
            "bom_pn":          bom_pn,
            "total_qty_needed": total_qty_needed,
        })

    # Consolidate COO, lifecycle
    consolidated_coo = "Unknown"
    for opt in all_options:
        if opt["coo"] not in ("Unknown","N/A",""):
            consolidated_coo = opt["coo"]
            break

    lifecycle_notes = ""
    for opt in all_options:
        if opt.get("eol"):         lifecycle_notes = "EOL"
        elif opt.get("discontinued"): lifecycle_notes = "DISC" if not lifecycle_notes else lifecycle_notes

    # Valid options (have pricing)
    valid_options = [
        o for o in all_options
        if (
            pd.notna(o.get("cost")) or
            o.get("stock", 0) > 0
            )]
    # Best cost option (lowest total cost)
    best_cost_option = min(valid_options, key=lambda o: o.get("cost", np.inf)) if valid_options else None

    # Fastest option (lowest effective lead time then cost)
    fastest_option = None
    if all_options:
        in_stock = [o for o in all_options if o.get("stock",0) >= total_qty_needed]
        if in_stock:
            fastest_option = min(in_stock, key=lambda o: o.get("cost", np.inf))
        else:
            with_lt = [o for o in all_options if o.get("lead_time", np.inf) != np.inf]
            if with_lt:
                fastest_option = min(with_lt, key=lambda o: o.get("lead_time", np.inf))

    total_stock = sum(o.get("stock",0) for o in all_options)

    # Risk scoring
    fastest_lt = fastest_option.get("lead_time", np.inf) if fastest_option else np.inf
    if isinstance(fastest_lt, float) and np.isinf(fastest_lt): fastest_lt_days = np.nan
    else: fastest_lt_days = fastest_lt

    risk_score, risk_factors = calculate_risk_score(
        sourcing_count   = len(valid_options),
        stock_available  = total_stock,
        qty_needed       = total_qty_needed,
        lead_time_days   = fastest_lt_days,
        lifecycle_notes  = lifecycle_notes,
        coo              = consolidated_coo,
    )

    tariff_rate = get_tariff_rate(consolidated_coo, custom_tariffs)

    status = "Active"
    if "EOL" in lifecycle_notes: status = "EOL"
    elif "DISC" in lifecycle_notes: status = "Discontinued"

    notes_list = []
    if total_stock < total_qty_needed: notes_list.append("Stock Gap")
    if best_cost_option and best_cost_option.get("notes"): notes_list.append(best_cost_option["notes"])

    # Best cost unit price with tariff
    bc_unit  = best_cost_option.get("unit_cost", np.nan) if best_cost_option else np.nan
    bc_total = best_cost_option.get("cost", np.nan) if best_cost_option else np.nan
    bc_qty   = best_cost_option.get("actual_order_qty","N/A") if best_cost_option else "N/A"
    bc_lt    = best_cost_option.get("lead_time", np.inf) if best_cost_option else np.inf
    bc_src   = best_cost_option.get("source","N/A") if best_cost_option else "N/A"

    desc = (best_cost_option or (all_options[0] if all_options else {})).get("Description","")

    return {
        "PartNumber":    bom_pn,
        "Manufacturer":  (best_cost_option or {}).get("Manufacturer", bom_mfg or "N/A"),
        "MfgPN":         (best_cost_option or {}).get("ManufacturerPartNumber", bom_pn),
        "QtyNeed":       total_qty_needed,
        "Status":        status,
        "Sources":       str(len(valid_options)),
        "StockAvail":    total_stock,
        "COO":           consolidated_coo,
        "TariffPct":     f"{tariff_rate*100:.1f}%",
        "TariffRate":    tariff_rate,
        "RiskScore":     risk_score,
        "RiskFactors":   risk_factors,
        "BestCostPer":   f"{bc_unit:.4f}" if pd.notna(bc_unit) else "N/A",
        "BestCostPerRaw": bc_unit,
        "BestTotalCost": (f"{bc_total:.2f}" if pd.notna(bc_total)else "Quote Required"),
        "BestTotalCostRaw": bc_total,
        "BestTotalWithTariff": (bc_total * (1 + tariff_rate)) if pd.notna(bc_total) else np.nan,
        "ActualBuyQty":  str(bc_qty),
        "BestCostLT":    f"{bc_lt:.0f}" if (pd.notna(bc_lt) and not np.isinf(bc_lt)) else ("0" if total_stock >= total_qty_needed else "N/A"),
        "BestCostSrc":   bc_src,
        "Description":   desc,
        "Notes":         "; ".join(notes_list),
        "DatasheetUrl":  (best_cost_option or {}).get("DatasheetUrl",""),
        "_options":      all_options,
        "_valid":        bool(valid_options),
    }


def calculate_strategies(part_results, config):
    """
    Port of calculate_summary_metrics strategy engine.
    Returns dict of strategy summaries.
    """
    total_units   = config.get("total_units", 100)
    target_lt     = config.get("target_lead_time_days", 56)
    max_premium   = config.get("max_premium", 15.0)
    cost_weight   = config.get("cost_weight", 0.5)
    lead_weight   = config.get("lead_time_weight", 0.5)
    buy_up_pct    = config.get("buy_up_threshold", 1.0)

    strategies = {
        "Lowest Cost (Strict)":      {"total_cost": 0.0, "max_lt": 0, "parts": {}, "invalid": False},
        "Lowest Cost (In Stock)":    {"total_cost": 0.0, "max_lt": 0, "parts": {}, "invalid": False},
        "Fastest Lead Time":         {"total_cost": 0.0, "max_lt": 0, "parts": {}, "invalid": False},
        "Optimized (Cost+LT)":       {"total_cost": 0.0, "max_lt": 0, "parts": {}, "invalid": False},
    }

    for part in part_results:
        if not part.get("_valid"): continue
        opts        = part["_options"]
        pn          = part["PartNumber"]
        qty_needed  = part["QtyNeed"]
        valid_opts  = [o for o in opts if pd.notna(o.get("cost")) and o.get("cost", np.inf) != np.inf]
        if not valid_opts: continue

        # ── Lowest Cost Strict ─────────────────────────────────────────────
        best = min(valid_opts, key=lambda o: o.get("cost", np.inf))
        strategies["Lowest Cost (Strict)"]["parts"][pn]  = best
        strategies["Lowest Cost (Strict)"]["total_cost"] += best.get("cost", 0)
        lt = best.get("lead_time", 0)
        if not (isinstance(lt, float) and np.isinf(lt)):
            strategies["Lowest Cost (Strict)"]["max_lt"] = max(strategies["Lowest Cost (Strict)"]["max_lt"], int(lt or 0))

        # ── Lowest Cost In-Stock ───────────────────────────────────────────
        in_stock = [o for o in valid_opts if o.get("stock",0) >= qty_needed]
        chosen   = min(in_stock, key=lambda o: o.get("cost", np.inf)) if in_stock else best
        strategies["Lowest Cost (In Stock)"]["parts"][pn]  = chosen
        strategies["Lowest Cost (In Stock)"]["total_cost"] += chosen.get("cost", 0)

        # ── Fastest Lead Time ──────────────────────────────────────────────
        def eff_lt(o): return 0 if o.get("stock",0) >= qty_needed else o.get("lead_time", np.inf)
        fastest = min(valid_opts, key=lambda o: (eff_lt(o), o.get("cost", np.inf)))
        strategies["Fastest Lead Time"]["parts"][pn]  = fastest
        strategies["Fastest Lead Time"]["total_cost"] += fastest.get("cost", 0)
        flt = eff_lt(fastest)
        if not (isinstance(flt, float) and np.isinf(flt)):
            strategies["Fastest Lead Time"]["max_lt"] = max(strategies["Fastest Lead Time"]["max_lt"], int(flt or 0))

        # ── Optimized (Cost + Lead Time) ───────────────────────────────────
        baseline_cost = best.get("cost", np.inf)
        constrained   = []
        for o in valid_opts:
            cost_o    = o.get("cost", np.inf)
            eff_lt_o  = eff_lt(o)
            if eff_lt_o == np.inf or eff_lt_o > target_lt: continue
            if baseline_cost > 1e-9:
                prem = (cost_o - baseline_cost) / baseline_cost * 100
            else:
                prem = 0
            if prem > max_premium: continue
            constrained.append(o)

        if constrained:
            costs = [safe_float(o.get("cost")) for o in constrained]
            lts   = [eff_lt(o) for o in constrained if eff_lt(o) != np.inf]
            min_c = min(costs) if costs else 0; max_c = max(costs) if costs else 1
            min_l = min(lts)   if lts   else 0; max_l = max(lts)   if lts   else 1
            c_rng = max(max_c - min_c, 1e-9); l_rng = max(max_l - min_l, 1e-9)
            best_score = np.inf; opt_chosen = None
            for o in constrained:
                nc    = (safe_float(o.get("cost")) - min_c) / c_rng
                nl    = (eff_lt(o) - min_l) / l_rng if eff_lt(o) != np.inf else 1.0
                score = cost_weight * nc + lead_weight * nl
                if o.get("eol") or o.get("discontinued"): score += 0.5
                if o.get("stock",0) < qty_needed: score += 0.1
                if score < best_score:
                    best_score = score; opt_chosen = o
        else:
            opt_chosen = fastest  # fallback

        strategies["Optimized (Cost+LT)"]["parts"][pn]  = opt_chosen or best
        strategies["Optimized (Cost+LT)"]["total_cost"] += (opt_chosen or best).get("cost", 0)
        olt = eff_lt(opt_chosen or best)
        if not (isinstance(olt, float) and np.isinf(olt)):
            strategies["Optimized (Cost+LT)"]["max_lt"] = max(strategies["Optimized (Cost+LT)"]["max_lt"], int(olt or 0))

    return strategies


import requests

def openai_ai_summary(data_context, openai_key, model="gpt-4o-mini"):
    """
    Call OpenAI API to generate executive AI summary.
    SSL verification disabled for corporate environments.
    """
    if not openai_key:
        return "⚠️ Add your OpenAI API key in the sidebar to enable AI summaries."

    system_prompt = (
        "You are a strategic supply chain advisor specializing in electronic components. "
        "Provide concise, actionable insights for executive review. "
        "Focus on risk, cost optimization, and build readiness."
    )

    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data_context},
        ],
        "max_tokens": 1200,
        "temperature": 0.6,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            verify=False,   # ✅ corporate SSL interception
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        elif response.status_code == 401:
            return "❌ Invalid OpenAI API key."

        elif response.status_code == 429:
            return "⚠️ Rate limit exceeded or insufficient OpenAI quota."

        else:
            return f"OpenAI API error {response.status_code}: {response.text}"

    except requests.exceptions.RequestException as e:
        return f"Error calling OpenAI: {e}"



# ── Streamlit Color Helpers ────────────────────────────────────────────────────

def color_risk_cell(val):
    if not isinstance(val, (int, float)): return ""
    if val >= 6.6:   return "background-color:#fee2e2; color:#900"
    elif val >= 3.6: return "background-color:#fef3c7; color:#7d3f00"
    return "background-color:#dcfce7; color:#155724"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 BOM Analyzer")
    st.caption("Web Edition v1.0.0 — PCB Department")
    st.divider()

    st.markdown("**🔑 Supplier API Keys**")
    mouser_key   = st.text_input("Mouser API Key",      type="password",
                                  placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    nexar_id     = st.text_input("Nexar Client ID",     type="password", placeholder="nexar.com")
    nexar_secret = st.text_input("Nexar Client Secret", type="password", placeholder="nexar.com")

    st.divider()
    openai_key = st.text_input("OpenAI API Key", type="password")

    # ── API Status Indicators ─────────────────────────────────────────────
    st.divider()
    st.markdown("**📡 API Status**")
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.markdown("🟢 Mouser" if mouser_key   else "⚫ Mouser")
    col_s2.markdown("🟢 Nexar"  if (nexar_id and nexar_secret) else "⚫ Nexar")
    col_s3.markdown("🟢 OpenAI"   if openai_key     else "⚫ OpenAI")
    st.caption("Keys are session-only and never stored.")

    st.divider()
    st.markdown("**🏗️ Build Configuration**")
    total_units     = st.number_input("Total Units to Build", min_value=1, value=100, step=10)
    target_lt_days  = st.number_input("Target Lead Time (days)", min_value=1, value=56, step=7,
                                       help="Maximum acceptable lead time for Optimized strategy")
    max_premium_pct = st.number_input("Max Cost Premium % (Optimized)", min_value=0.0, value=15.0, step=1.0,
                                       help="How much more expensive than cheapest option is acceptable")
    cost_w  = st.slider("Cost Weight",      0.0, 1.0, 0.50, 0.05)
    lead_w  = st.slider("Lead Time Weight", 0.0, 1.0, 0.50, 0.05)
    buy_up  = st.number_input("Buy-Up Threshold %", min_value=0.0, value=1.0, step=0.5,
                               help="Allow buying to next price break if cost increase is within this %")

    st.divider()
    st.markdown("**🌍 Custom Tariff Rates (%)**")
    st.caption("Leave blank to use defaults (China 25%, others 3.5%)")
    custom_tariffs = {}
    tariff_countries = ["China","Mexico","India","Vietnam","Taiwan","Japan","Malaysia",
                        "Germany","USA","Philippines","Thailand","South Korea"]
    cols_t = st.columns(2)
    for i, country in enumerate(tariff_countries):
        with cols_t[i % 2]:
            rate_str = st.text_input(country, value="", key=f"tariff_{country}", label_visibility="visible")
            if rate_str.strip():
                r = safe_float(rate_str)
                if pd.notna(r) and r >= 0:
                    custom_tariffs[country] = r / 100.0

config = {
    "total_units":           total_units,
    "target_lead_time_days": target_lt_days,
    "max_premium":           max_premium_pct,
    "cost_weight":           cost_w,
    "lead_time_weight":      lead_w,
    "buy_up_threshold":      buy_up,
    "custom_tariff_rates":   custom_tariffs,
}

# ── Main Page ─────────────────────────────────────────────────────────────────
st.markdown('<div class="title-bar">🔬 BOM Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Supply Chain BOM Optimizer · Risk Scoring · AI-Powered Insights (OpenAI)</div>', unsafe_allow_html=True)

# ── BOM Upload ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">📂 Step 1 — Upload Your BOM</div>', unsafe_allow_html=True)

col_up, col_tmpl = st.columns([3,1])
with col_up:
    uploaded = st.file_uploader("Upload BOM CSV", type=["csv"],
        help="Must have 'Part Number' and 'Quantity' columns. 'Manufacturer' and 'Description' optional.")
with col_tmpl:
    template = pd.DataFrame({
        "Part Number":  ["LM358DR","RMCF0402FT100K","GRM188R71C104KA01D"],
        "Quantity":     [2,10,4],
        "Manufacturer": ["Texas Instruments","Stackpole","Murata"],
        "Description":  ["Op-Amp Dual","Resistor 100K 0402","Cap 100nF 0402"],
    })
    st.download_button("⬇️ BOM Template", template.to_csv(index=False),
                       "bom_template.csv","text/csv", use_container_width=True)

if uploaded:
    try:
        raw_df = pd.read_csv(uploaded, skipinitialspace=True, on_bad_lines='skip')
        # Collapse any embedded newlines inside fields (common Excel CSV export artifact)
        raw_df = raw_df.apply(lambda col: col.map(lambda v: str(v).replace("\n"," ").replace("\r","").strip() if isinstance(v, str) else v))
        raw_df.columns = [c.strip() for c in raw_df.columns]

        # Normalize columns (same logic as source startup guide)
        col_map = {}
        for c in raw_df.columns:
            cl = c.lower().replace(" ","").replace("_","").replace(".","")
            if cl in ["partnumber","pn","mpn","partno","partnum"]:   col_map[c]="Part Number"
            elif cl in ["quantity","qty","q","amount","qtyperunit"]: col_map[c]="Quantity"
            elif cl in ["manufacturer","mfg","mfr"]:                 col_map[c]="Manufacturer"
            elif cl in ["description","desc","partdescription"]:     col_map[c]="Description"
        raw_df.rename(columns=col_map, inplace=True)

        if "Part Number" not in raw_df.columns or "Quantity" not in raw_df.columns:
            st.error("❌ CSV must have 'Part Number' and 'Quantity' columns.")
            st.stop()

        raw_df["Quantity"]     = pd.to_numeric(raw_df["Quantity"], errors="coerce").fillna(1).astype(int)
        raw_df["Part Number"]  = raw_df["Part Number"].astype(str).str.strip()
        raw_df["Manufacturer"] = raw_df.get("Manufacturer", pd.Series([""] * len(raw_df))).fillna("").astype(str)
        raw_df = raw_df[raw_df["Part Number"].str.len() > 0].dropna(subset=["Part Number"])

        # ── Auto-clean part numbers & report changes ───────────────────────
        cleaned_log = []
        def apply_clean(pn):
            cleaned, original, changes = clean_part_number(pn)
            if changes:
                cleaned_log.append({"Original": original, "Cleaned To": cleaned, "Changes": ", ".join(changes)})
            return cleaned
        raw_df["Part Number"] = raw_df["Part Number"].apply(apply_clean)
        if cleaned_log:
            with st.expander(f"🔧 Auto-cleaned {len(cleaned_log)} part number(s) — click to review", expanded=True):
                st.caption("These part numbers had formatting issues (apostrophes, suffixes) that were automatically fixed before sending to supplier APIs.")
                st.dataframe(pd.DataFrame(cleaned_log), use_container_width=True, hide_index=True)

        st.success(f"✅ BOM loaded: **{len(raw_df)} parts**, {raw_df['Quantity'].sum()} total component placements")
        with st.expander("👁 Preview BOM", expanded=False):
            st.dataframe(raw_df, use_container_width=True)

        st.divider()
        st.markdown('<div class="section-head">🚀 Step 2 — Run Analysis</div>', unsafe_allow_html=True)

        if not (mouser_key or (nexar_id and nexar_secret)):
            st.warning("⚠️ No supplier API keys entered — results will show 'Not Found'. "
                       "Add Mouser or Nexar keys in the sidebar for live pricing data.")

        run_btn = st.button("▶️ Run BOM Analysis", type="primary", use_container_width=True)

        if run_btn:
            st.session_state.pop("results", None)
            st.session_state.pop("strategies", None)
            st.session_state.pop("ai_summary", None)

            nexar_token_cache = {}
            results = []
            progress_bar = st.progress(0, text="Starting analysis...")
            status_txt   = st.empty()
            total_parts  = len(raw_df)

            for i, row in raw_df.iterrows():
                pn   = str(row["Part Number"]).strip()
                qty  = int(row["Quantity"])
                mfg  = str(row.get("Manufacturer","")).strip()
                status_txt.text(f"🔍 {pn}  ({len(results)+1}/{total_parts})")
                progress_bar.progress((len(results)+1)/total_parts, text=f"Analyzing {pn}…")

                result = analyze_single_part(pn, mfg, qty, config,
                                             mouser_key, nexar_id, nexar_secret, nexar_token_cache)
                results.append(result)
                time.sleep(0.1)  # polite rate-limiting

            progress_bar.empty(); status_txt.empty()
            st.session_state["results"]    = results
            st.session_state["strategies"] = calculate_strategies(results, config)
            st.success(f"✅ Analysis complete — {len(results)} parts processed")

        # ── Results Display ────────────────────────────────────────────────
        if "results" in st.session_state:
            results    = st.session_state["results"]
            strategies = st.session_state["strategies"]

            valid_results = [r for r in results if r.get("_valid")]

            # KPI metrics
            total_cost_best   = sum(v for r in valid_results if pd.notna(v := r.get("BestTotalCostRaw") or 0))
            total_cost_tariff = sum(v for r in valid_results if pd.notna(v := r.get("BestTotalWithTariff") or 0))
            tariff_impact     = total_cost_tariff - total_cost_best
            high_risk   = sum(1 for r in results if r.get("RiskScore",0) >= 6.6)
            mod_risk    = sum(1 for r in results if 3.6 <= r.get("RiskScore",0) < 6.6)
            low_risk    = sum(1 for r in results if r.get("RiskScore",0) < 3.6)
            eol_count   = sum(1 for r in results if r.get("Status") in ("EOL","Discontinued"))
            no_stock    = sum(1 for r in results if r.get("StockAvail",0) == 0)
            not_found   = sum(1 for r in results if not r.get("_valid"))

            st.divider()
            st.markdown('<div class="section-head">📊 Results</div>', unsafe_allow_html=True)

            k1,k2,k3,k4,k5,k6 = st.columns(6)
            k1.metric("Total BOM Cost",       f"${total_cost_best:,.2f}")
            k2.metric("Cost with Tariffs",     f"${total_cost_tariff:,.2f}", delta=f"+${tariff_impact:,.2f}")
            k3.metric("🔴 High Risk",          high_risk)
            k4.metric("🟡 Moderate Risk",      mod_risk)
            k5.metric("🟢 Low Risk",           low_risk)
            k6.metric("❌ Not Found / EOL",    f"{not_found + eol_count}")

            # ── Not Found Diagnostic ──────────────────────────────────────
            not_found_parts = [r for r in results if not r.get("_valid")]
            if not_found_parts:
                with st.expander(f"⚠️ {len(not_found_parts)} part(s) returned no supplier data — click for guidance", expanded=True):
                    st.markdown("""
**Common reasons a part returns no data:**
- Part number has a distributor suffix (e.g. `2N3906 PBFREE` → try `2N3906`)
- Part number starts with an apostrophe from Excel export (auto-fixed on next run)
- Part is too new, too old, or niche for Mouser's catalog
- Part number belongs to a manufacturer not stocked by Mouser (try Nexar)
- Mouser daily API limit reached (1,000 calls/day free tier)
                    """)
                    nf_rows = [{"Part Number": r["PartNumber"],
                                "Total Qty Needed": r["QtyNeed"],
                                "Suggestion": "Try searching manually on mouser.com or digikey.com"
                                } for r in not_found_parts]
                    st.dataframe(pd.DataFrame(nf_rows), use_container_width=True, hide_index=True)
                    st.caption("These parts still appear in the BOM Analysis table with Risk Score = 10 (no data = maximum risk).")

            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📋 BOM Analysis", "💰 Strategies", "📈 Visualizations", "🤖 AI Summary"])

            # ── Tab 1: Full Results ────────────────────────────────────────
            with tab1:
                display_rows = []
                for r in results:
                    display_rows.append({
                        "Part Number":    r["PartNumber"],
                        "Description":    r.get("Description","")[:60],
                        "BOM Qty":        r["QtyNeed"] // total_units,
                        "Total Qty":      r["QtyNeed"],
                        "Sources":        r["Sources"],
                        "Best Supplier":  r["BestCostSrc"],
                        "Unit Cost ($)":  r.get("BestCostPerRaw", np.nan),
                        "Total Cost ($)": r.get("BestTotalCostRaw", np.nan),
                        "w/Tariff ($)":   r.get("BestTotalWithTariff", np.nan),
                        "Tariff":         r["TariffPct"],
                        "Stock":          r["StockAvail"],
                        "Lead (days)":    r["BestCostLT"],
                        "COO":            r["COO"],
                        "Status":         r["Status"],
                        "Risk Score":     r["RiskScore"],
                        "Notes":          r.get("Notes",""),
                    })
                res_df = pd.DataFrame(display_rows)

                # Risk filter
                risk_filter = st.radio("Filter by Risk:", ["All","🔴 High","🟡 Moderate","🟢 Low"],
                                        horizontal=True, key="risk_filter_tab1")
                if risk_filter == "🔴 High":
                    res_df = res_df[res_df["Risk Score"] >= 6.6]
                elif risk_filter == "🟡 Moderate":
                    res_df = res_df[(res_df["Risk Score"] >= 3.6) & (res_df["Risk Score"] < 6.6)]
                elif risk_filter == "🟢 Low":
                    res_df = res_df[res_df["Risk Score"] < 3.6]

                styled = res_df.style\
                    .map(color_risk_cell, subset=["Risk Score"])\
                    .format({
                        "Unit Cost ($)":  lambda v: f"${v:.4f}" if pd.notna(v) else "N/A",
                        "Total Cost ($)": lambda v: f"${v:,.2f}" if pd.notna(v) else "N/A",
                        "w/Tariff ($)":   lambda v: f"${v:,.2f}" if pd.notna(v) else "N/A",
                        "Risk Score":     lambda v: f"{v:.1f}" if pd.notna(v) else "N/A",
                    })
                st.dataframe(styled, use_container_width=True, height=500)

                # Risk factor breakdown
                with st.expander("🔍 Risk Factor Details per Part"):
                    rf_rows = []
                    for r in sorted(results, key=lambda x: x.get("RiskScore",0), reverse=True):
                        rf = r.get("RiskFactors", {})
                        rf_rows.append({
                            "Part Number": r["PartNumber"],
                            "Overall Risk": r["RiskScore"],
                            "Sourcing":    rf.get("Sourcing",""),
                            "Stock":       rf.get("Stock",""),
                            "Lead Time":   rf.get("LeadTime",""),
                            "Lifecycle":   rf.get("Lifecycle",""),
                            "Geographic":  rf.get("Geographic",""),
                            "Status":      r["Status"],
                            "COO":         r["COO"],
                        })
                    rf_df = pd.DataFrame(rf_rows)
                    st.dataframe(rf_df.style.map(color_risk_cell, subset=["Overall Risk"]),
                                 use_container_width=True)

                # Export
                export_df = pd.DataFrame([{
                    "Part Number":    r["PartNumber"],
                    "Manufacturer":   r["Manufacturer"],
                    "MfgPN":          r["MfgPN"],
                    "Description":    r.get("Description",""),
                    "BOM Qty":        r["QtyNeed"] // total_units,
                    "Total Qty Needed": r["QtyNeed"],
                    "Best Supplier":  r["BestCostSrc"],
                    "Unit Cost ($)":  r.get("BestCostPerRaw",""),
                    "Total Cost ($)": r.get("BestTotalCostRaw",""),
                    "Total w/Tariff ($)": r.get("BestTotalWithTariff",""),
                    "Tariff Rate":    r["TariffPct"],
                    "Actual Buy Qty": r["ActualBuyQty"],
                    "Stock Available": r["StockAvail"],
                    "Lead Time (days)": r["BestCostLT"],
                    "COO":            r["COO"],
                    "Status":         r["Status"],
                    "Risk Score":     r["RiskScore"],
                    "Sourcing Risk":  r.get("RiskFactors",{}).get("Sourcing",""),
                    "Stock Risk":     r.get("RiskFactors",{}).get("Stock",""),
                    "LeadTime Risk":  r.get("RiskFactors",{}).get("LeadTime",""),
                    "Lifecycle Risk": r.get("RiskFactors",{}).get("Lifecycle",""),
                    "Geographic Risk": r.get("RiskFactors",{}).get("Geographic",""),
                    "Datasheet":      r.get("DatasheetUrl",""),
                    "Notes":          r.get("Notes",""),
                } for r in results])
                st.download_button("⬇️ Export Full BOM Analysis CSV",
                    export_df.to_csv(index=False),
                    f"BOM_Analysis_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    "text/csv", use_container_width=True)

            # ── Tab 2: Purchasing Strategies ───────────────────────────────
            with tab2:
                st.markdown("Compare the 4 purchasing strategies from the original BOM Analyzer.")
                strat_summary = []
                for sname, sdata in strategies.items():
                    strat_summary.append({
                        "Strategy":       sname,
                        "Total BOM Cost": f"${sdata['total_cost']:,.2f}",
                        "Max Lead Time":  f"{sdata['max_lt']} days",
                        "Parts Covered":  len(sdata["parts"]),
                    })
                st.dataframe(pd.DataFrame(strat_summary), use_container_width=True, hide_index=True)

                chosen_strat = st.selectbox("📋 View / Export Strategy Details:",
                                            list(strategies.keys()))
                strat_parts = strategies[chosen_strat]["parts"]
                strat_rows  = []
                for pn, opt in strat_parts.items():
                    lt_val = opt.get("lead_time", np.inf)
                    lt_str = f"{lt_val:.0f}" if (pd.notna(lt_val) and not np.isinf(lt_val)) else "In Stock / N/A"
                    strat_rows.append({
                        "Part Number":   pn,
                        "Supplier":      opt.get("source","N/A"),
                        "Unit Cost ($)": opt.get("unit_cost", np.nan),
                        "Total Cost ($)": opt.get("cost", np.nan),
                        "Qty Order":     opt.get("actual_order_qty","N/A"),
                        "Stock":         opt.get("stock",0),
                        "Lead (days)":   lt_str,
                        "Notes":         opt.get("notes",""),
                    })
                strat_df = pd.DataFrame(strat_rows)
                st.dataframe(strat_df.style.format({
                    "Unit Cost ($)":  lambda v: f"${v:.4f}" if pd.notna(v) else "N/A",
                    "Total Cost ($)": lambda v: f"${v:,.2f}" if pd.notna(v) else "N/A",
                }), use_container_width=True, height=450)

                strat_export = strat_df.copy()
                st.download_button(f"⬇️ Export '{chosen_strat}' Strategy CSV",
                    strat_export.to_csv(index=False),
                    f"Strategy_{chosen_strat.replace(' ','_')}_{datetime.now():%Y%m%d_%H%M}.csv",
                    "text/csv", use_container_width=True)

            # ── Tab 3: Visualizations ──────────────────────────────────────
            with tab3:
                import matplotlib.pyplot as plt

                chart_type = st.selectbox("Select Chart:", [
                    "Risk Score Distribution",
                    "Top Parts by Cost",
                    "Stock vs Qty Needed",
                    "Cost + Tariff Impact (Top 15)",
                    "COO Geographic Risk Map",
                    "Strategy Cost Comparison",
                ])
                fig, ax = plt.subplots(figsize=(11,5))
                fig.patch.set_facecolor("#f8f9fa"); ax.set_facecolor("#f8f9fa")

                if chart_type == "Risk Score Distribution":
                    bins   = [0, 3.5, 6.5, 10]
                    labels = ["Low (0-3.5)", "Moderate (3.6-6.5)", "High (6.6-10)"]
                    colors = ["#107c10","#ca5010","#d13438"]
                    counts = [
                        sum(1 for r in results if r.get("RiskScore",0) <= 3.5),
                        sum(1 for r in results if 3.5 < r.get("RiskScore",0) <= 6.5),
                        sum(1 for r in results if r.get("RiskScore",0) > 6.5),
                    ]
                    bars = ax.bar(labels, counts, color=colors, width=0.5)
                    for bar,v in zip(bars,counts):
                        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, str(v),
                                ha="center", fontweight="bold", fontsize=12)
                    ax.set_ylabel("Number of Parts"); ax.set_title("Risk Score Distribution")

                elif chart_type == "Top Parts by Cost":
                    top = sorted(valid_results, key=lambda r: r.get("BestTotalCostRaw",0) or 0, reverse=True)[:20]
                    ax.barh([r["PartNumber"] for r in top],
                            [r.get("BestTotalCostRaw",0) or 0 for r in top], color="#0078d4")
                    ax.set_xlabel("Extended Cost ($)"); ax.set_title("Top 20 Parts by Cost")

                elif chart_type == "Stock vs Qty Needed":
                    scores  = [r.get("RiskScore",0) for r in results]
                    x_vals  = [r.get("QtyNeed",0) for r in results]
                    y_vals  = [r.get("StockAvail",0) for r in results]
                    sc = ax.scatter(x_vals, y_vals, c=scores, cmap="RdYlGn_r",
                                   s=80, alpha=0.75, vmin=0, vmax=10)
                    mx = max(max(x_vals,default=1), max(y_vals,default=1))*1.1
                    ax.plot([0,mx],[0,mx],"k--",alpha=0.4,label="Stock = Needed")
                    plt.colorbar(sc, ax=ax, label="Risk Score")
                    ax.set_xlabel("Qty Needed"); ax.set_ylabel("Stock Available")
                    ax.set_title("Stock vs Quantity Needed")
                    ax.legend()

                elif chart_type == "Cost + Tariff Impact (Top 15)":
                    top = sorted(valid_results, key=lambda r: r.get("BestTotalCostRaw",0) or 0, reverse=True)[:15]
                    pns  = [r["PartNumber"] for r in top]
                    base = [r.get("BestTotalCostRaw",0) or 0 for r in top]
                    tariff_add = [(r.get("BestTotalWithTariff",0) or 0) - (r.get("BestTotalCostRaw",0) or 0) for r in top]
                    x = range(len(pns))
                    ax.bar(x, base, label="Base Cost", color="#0078d4")
                    ax.bar(x, tariff_add, bottom=base, label="Tariff Add-on", color="#d13438", alpha=0.8)
                    ax.set_xticks(list(x)); ax.set_xticklabels(pns, rotation=45, ha="right", fontsize=8)
                    ax.set_ylabel("Cost ($)"); ax.set_title("Base Cost vs Tariff Impact")
                    ax.legend()

                elif chart_type == "COO Geographic Risk Map":
                    coo_risk = {}
                    for r in results:
                        coo = r.get("COO","Unknown")
                        geo = r.get("RiskFactors",{}).get("Geographic", GEO_RISK_TIERS.get("_DEFAULT_",4))
                        coo_risk[coo] = max(coo_risk.get(coo,0), geo)
                    coos  = list(coo_risk.keys())
                    risks = list(coo_risk.values())
                    colors= ["#d13438" if v>=7 else "#ca5010" if v>=4 else "#107c10" for v in risks]
                    ax.barh(coos, risks, color=colors)
                    ax.set_xlabel("Geographic Risk Score (0-10)")
                    ax.set_title("Geographic Risk by Country of Origin")
                    ax.axvline(x=5, color="orange", linestyle="--", alpha=0.6, label="Moderate threshold")
                    ax.axvline(x=7, color="red",    linestyle="--", alpha=0.6, label="High threshold")
                    ax.legend(fontsize=8)

                elif chart_type == "Strategy Cost Comparison":
                    names  = list(strategies.keys())
                    totals = [strategies[n]["total_cost"] for n in names]
                    colors = ["#0078d4","#107c10","#ca5010","#d13438"]
                    bars   = ax.bar(names, totals, color=colors[:len(names)], width=0.5)
                    for bar, v in zip(bars,totals):
                        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                                f"${v:,.0f}", ha="center", fontsize=9, fontweight="bold")
                    ax.set_ylabel("Total BOM Cost ($)"); ax.set_title("Purchasing Strategy Cost Comparison")
                    ax.set_xticklabels(names, rotation=10, ha="right")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # ── Tab 4: AI Summary ──────────────────────────────────────────
            with tab4:
                st.markdown("### 🤖 AI Executive Summary")
                st.caption(f"Powered by **OpenAI** — {openai_key}")

                if not openai_key:
                    st.warning("Add your OpenAI API key in the sidebar. ")
                else:
                    # Build a rich prompt context matching original app's AI logic
                    high_risk_parts = [r for r in results if r.get("RiskScore",0) >= 6.6]
                    eol_parts       = [r for r in results if r.get("Status") in ("EOL","Discontinued")]
                    stock_gap_parts = [r for r in results if r.get("StockAvail",0) < r.get("QtyNeed",0)]
                    no_price_parts  = [r for r in results if not r.get("_valid")]

                    critical_detail = ""
                    for r in high_risk_parts[:8]:
                        critical_detail += (f"\n  - {r['PartNumber']}: Risk={r['RiskScore']}, "
                                            f"Stock={r['StockAvail']}/{r['QtyNeed']} needed, "
                                            f"LT={r['BestCostLT']} days, Status={r['Status']}, COO={r['COO']}")

                    strat_summary_text = ""
                    for sname, sdata in strategies.items():
                        strat_summary_text += f"\n  - {sname}: ${sdata['total_cost']:,.2f} total, max LT {sdata['max_lt']} days"

                    prompt = f"""Analyze this BOM for a PCB electronics manufacturing team building {total_units} units.

SUMMARY METRICS:
- Total Parts: {len(results)}
- Valid (with pricing): {len(valid_results)}
- Not Found / No Data: {len(no_price_parts)}
- Total BOM Cost (best price): ${total_cost_best:,.2f}
- Total BOM Cost (with tariffs): ${total_cost_tariff:,.2f}
- Tariff Impact: ${tariff_impact:,.2f}
- High Risk Parts (≥6.6): {high_risk}
- Moderate Risk Parts (3.6–6.5): {mod_risk}
- Low Risk Parts (<3.6): {low_risk}
- EOL/Discontinued Parts: {eol_count}
- Parts with Zero Stock: {no_stock}
- Parts with Stock Gaps: {len(stock_gap_parts)}

PURCHASING STRATEGIES:
{strat_summary_text}

HIGH RISK PARTS DETAIL (risk ≥6.6):
{critical_detail if critical_detail else 'None'}

EOL / DISCONTINUED:
{", ".join(r["PartNumber"] for r in eol_parts[:10]) or "None"}

PARTS WITH STOCK GAPS:
{", ".join(r["PartNumber"] for r in stock_gap_parts[:10]) or "None"}

Please provide:
1. **Executive Summary** (2-3 sentences)
2. **Critical Risks** — specific parts needing immediate attention
3. **Top 3 Procurement Recommendations** — actionable steps
4. **Cost Optimization Opportunities**
5. **Recommended Purchasing Strategy** and why

Be specific, concise, and actionable. Reference actual part numbers where relevant."""

                    if st.button("🤖 Generate AI Summary", type="primary"):
                        with st.spinner("OpenAI is analyzing your BOM..."):
                            summary = openai_ai_summary(prompt, openai_key)
                            st.session_state["ai_summary"] = summary

                    if "ai_summary" in st.session_state:
                        st.markdown(st.session_state["ai_summary"])
                        st.download_button("⬇️ Export AI Summary",
                            st.session_state["ai_summary"],
                            f"AI_Summary_{datetime.now():%Y%m%d_%H%M%S}.txt",
                            "text/plain")

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback; st.code(traceback.format_exc())

else:
    # Welcome / instructions
    st.info("👆 Upload a BOM CSV file above to get started. Download the template if you need the format.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**What this tool analyzes:**
- 📡 Real-time pricing, stock & lead times from **Mouser** and **Nexar/Octopart**
- 💰 Optimal purchase quantities using **price break + buy-up logic** (exact original algorithm)
- ⚠️ Multi-factor **risk scoring** (Sourcing, Stock, Lead Time, Lifecycle, Geographic) — same weights as desktop app
- 🌍 **Tariff/duty estimation** by country of origin
- 📊 **4 purchasing strategies**: Lowest Cost (Strict), Lowest Cost (In Stock), Fastest, Optimized
- 🤖 **AI executive summary** via Groq (free, no credit card)
- 📤 Full **CSV export** for every result and strategy
        """)
    with col_b:
        st.markdown("""
**CSV Format Required:**
```
Part Number, Quantity, Manufacturer, Description
LM358DR, 2, Texas Instruments, Op-Amp Dual
RMCF0402FT100K, 10, Stackpole, Resistor 100K
GRM188R71C104KA01D, 4, Murata, Cap 100nF
```
`Part Number` and `Quantity` are required.
`Manufacturer` and `Description` are optional.

**API Keys Needed :**
- 🔑 [Mouser API](https://www.mouser.com/api-search/) — mouser.com
- 🔑 [Nexar API](https://nexar.com) — nexar.com
- 🤖 [OpenAI](https://openai.com/api/) — for AI summary
        """)

st.divider()
st.caption("BOM Analyzer Web Edition · CRDV Adaptation Initiative · AI by OpenAI · For PCB Department use")