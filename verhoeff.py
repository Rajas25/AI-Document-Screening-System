# Complete and accurate Verhoeff Algorithm for Aadhaar Validation

verhoeff_table_d = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

verhoeff_table_p = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

verhoeff_table_inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]

def validate_aadhaar(aadhaar_num: str) -> bool:
    # Clean spaces, hyphens, or dots from the input string text
    clean_num = "".join(str(aadhaar_num).split()).replace("-", "").replace(".", "")
    
    # An Aadhaar sequence must be exactly 12 numeric digits
    if not clean_num.isdigit() or len(clean_num) != 12:
        return False
    
    # Calculate array validation sum properties
    c = 0
    for i, item in enumerate(reversed(clean_num)):
        c = verhoeff_table_d[c][verhoeff_table_p[i % 8][int(item)]]
        
    return c == 0
