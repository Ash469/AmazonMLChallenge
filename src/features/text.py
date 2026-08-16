import re
import pandas as pd

def clean_text(text):
    """
    Cleans text sequence: handles NaNs, lowercase conversion, and strip formatting.
    Preserves dimensions, units, hyphens, and alphanumeric codes.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Strip excess whitespaces
    text = text.strip()
    
    # Normalize unicode symbols (like x/×/X to standard 'x')
    text = re.sub(r'[\u00d7\u2715]', ' x ', text)
    
    return text

def get_combined_text(df: pd.DataFrame):
    """
    Combines cleaned TITLE, BULLET_POINTS, and DESCRIPTION columns into a single string.
    """
    titles = df['TITLE'].apply(clean_text)
    bullets = df['BULLET_POINTS'].apply(clean_text)
    descriptions = df['DESCRIPTION'].apply(clean_text)
    
    combined = titles + " " + bullets + " " + descriptions
    return combined.str.strip()
