from enum import Enum

class ClassificationLabel(Enum):
    HATE = ("Hate", "#ff4444")
    TOXIC = ("Toxic", "#ff6b35")
    OFFENSIVE = ("Offensive", "#ffa500")
    NEUTRAL = ("Neutral", "#28a745")
    AMBIGUOUS = ("Ambiguous", "#6c757d")
    
    def __init__(self, label, color):
        self.label = label
        self.color = color
