import re
import pandas as pd
import numpy as np

# convert standard units to "inches * 100" (target unit)
UNIT_TO_TARGET_INCH_100 = {
    'in': 100.0, 'inch': 100.0, 'inches': 100.0, '"': 100.0,
    'cm': 100.0 / 2.54, 'centimeter': 100.0 / 2.54, 'centimeters': 100.0 / 2.54,
    'mm': 100.0 / 25.4, 'millimeter': 100.0 / 25.4, 'millimeters': 100.0 / 25.4,
    'm': 100.0 * 100.0 / 2.54, 'meter': 100.0 * 100.0 / 2.54, 'meters': 100.0 * 100.0 / 2.54,
    'ft': 1200.0, 'foot': 1200.0, 'feet': 1200.0,
    'yd': 3600.0, 'yard': 3600.0, 'yards': 3600.0
}

# Dimension pattern matching (e.g. 10 x 20 x 30 inches, 15.6 inch, etc.)
DIM_PATTERN = re.compile(
    r'(?:\b\d+(?:[\.,]\d+)?\s*(?:x|×|X)\s*)*\b\d+(?:[\.,]\d+)?\s*(?:inches|inch|in|cm|mm|m|ft|foot|feet|yd|yard|yards|[\"\'])\b',
    re.IGNORECASE
)

# Ignore values directly adjacent to non-dimension units
IGNORE_PATTERN = re.compile(r'\b(?:mah|gb|w|hz|v|gsm|pcs|pack|set|dpi|lbs|kg|g|oz|ml)\b', re.IGNORECASE)

def parse_text_dimensions(text: str):
    """
    Scans a text string for dimension measurements and returns values converted to target scale (inches * 100).
    Returns dictionary of dimensional statistics:
      - 'dim_max': maximum parsed value
      - 'dim_min': minimum parsed value
      - 'dim_mean': mean parsed value
      - 'dim_count': count of parsed dimension components
    """
    if not text:
        return {'dim_max': 0.0, 'dim_min': 0.0, 'dim_mean': 0.0, 'dim_count': 0}
        
    matches = DIM_PATTERN.findall(text)
    if not matches:
        return {'dim_max': 0.0, 'dim_min': 0.0, 'dim_mean': 0.0, 'dim_count': 0}
        
    extracted_values = []
    for match in matches:
        # Check ignore patterns
        if IGNORE_PATTERN.search(match):
            continue
            
        # Identify unit type
        unit_match = re.search(r'(inches|inch|in|cm|mm|m|ft|foot|feet|yd|yard|yards|[\"\'])', match, re.IGNORECASE)
        if not unit_match:
            continue
        unit = unit_match.group(1).lower()
        conv = UNIT_TO_TARGET_INCH_100.get(unit, 100.0)
        
        # Get individual numeric matches
        numbers = re.findall(r'\b\d+(?:[\.,]\d+)?', match)
        for num_str in numbers:
            try:
                num = float(num_str.replace(',', '.'))
                val_target = num * conv
                # Exclude obvious outlier numbers
                if 0.1 < val_target < 100000.0:
                    extracted_values.append(val_target)
            except ValueError:
                continue
                
    if not extracted_values:
        return {'dim_max': 0.0, 'dim_min': 0.0, 'dim_mean': 0.0, 'dim_count': 0}
        
    return {
        'dim_max': float(np.max(extracted_values)),
        'dim_min': float(np.min(extracted_values)),
        'dim_mean': float(np.mean(extracted_values)),
        'dim_count': len(extracted_values)
    }

def extract_dimension_features(texts: pd.Series):
    """
    Applies parsing over a Series of texts and returns a DataFrame of dimension features.
    """
    parsed_dicts = texts.apply(parse_text_dimensions)
    return pd.DataFrame(list(parsed_dicts))
