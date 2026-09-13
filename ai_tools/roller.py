import random
import statistics

def roll(sides: int, count: int = 1, modifier: int = 0) -> str:
    rolls = [random.randint(1, sides) for _ in range(count)]
    
    total = sum(rolls)
    lowest = min(rolls)
    highest = max(rolls)
    
    mean = statistics.mean(rolls)
    med = statistics.median(rolls)
    var = statistics.variance(rolls, mean) if count > 1 else 0.0
    mode = statistics.mode(rolls)

    return (
        "Roll Results: "
        f"Total: {total + modifier}, Lowest Roll: {lowest + modifier}, Highest Roll: {highest + modifier}, "
        f"Mean: {mean + modifier:.2f}, Median: {med + modifier}, Mode: {mode + modifier}, Variance: {var:.2f}"
    )