from enum import StrEnum
import itertools
import random
import secrets

class Styles(StrEnum):
    COIN = "coin flip"
    DIE = "dice roll"
    POKER_HAND = "poker hand"
    TAROT_READING = "tarot reading"
    SECRETS = "secret token"

POKER_DECK = tuple(
    f'{rank} of {suit}' 
    for rank, suit in itertools.product(
        ("Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King"),
        ("Diamonds", "Hearts", "Spades", "Clubs")
    )
)

TAROT_DECK = tuple([
    f'{rank} of {suit}' 
    for rank, suit in itertools.product(
        ("Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Page", "Knight", "Queen", "King"),
        ("Cups", "Pentacles", "Wands", "Swords")
    )
] + [
    "0. The Fool", "I. The Magician", "II. The High Priestess", "III. The Empress", "IV. The Emperor", "V. The Hierophant", "VI. The Lovers",
    "VII. The Chariot", "VIII. Justice", "IX. The Hermit", "X. Wheel of Fortune", "XI. Strength", "XII. The Hanged Man", "XIII. Death", "XIV. Temperance", 
    "XV. The Devil", "XVI. The Tower", "XVII. The Star", "XVIII. The Moon", "XIX. The Sun", "XX. Judgment", "XXI. The World"
])

def _coin() -> str:
    return f"You flipped a: {'Heads' if random.random() < 0.5 else 'Tails'}"
    
def _die() -> str:
    return f"You rolled a: {random.randint(1, 20)}"
    
def _poker_hand() -> str:
    return f"You drew: {', '.join(random.sample(POKER_DECK, 5))}"

def _tarot_reading() -> str:
    # Shuffle and copy reference constant in one go
    deck = random.sample(TAROT_DECK, 78)
    
    # Simulate humanistic deck cut
    cut_i = int(random.gauss(37, 8))
    cut_i = max(15, min(63, cut_i))
    
    # Reverse card orientation at cut
    deck = deck[:cut_i] + [f"{card} (Reversed)" for card in deck[cut_i:]] 
    
    # Re-shuffle
    random.shuffle(deck)
    
    # Deal the fates
    return f"Past: {deck[0]}, Present: {deck[1]}, Future: {deck[2]}"

def _secrets() -> str:
    return f'Secret Token: {secrets.token_urlsafe(16)}'

def randomizer(style: Styles) -> str:
    match style:
        case Styles.COIN:
            return _coin()
        case Styles.DIE:
            return _die()
        case Styles.POKER_HAND:
            return _poker_hand()
        case Styles.TAROT_READING:
            return _tarot_reading()
        case Styles.SECRETS:
            return _secrets()
        case _:
            return "ERROR: Not a valid mode of randomness."