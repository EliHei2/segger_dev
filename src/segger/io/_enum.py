from enum import Enum

class Filter_Substrings(Enum):
    tenx_xenium = [
        'NegControlProbe_*',
        'antisense_*',
        'NegControlCodeword*',
        'BLANK_*',
        'DeprecatedCodeword_*',
        'UnassignedCodeword_*',
    ]