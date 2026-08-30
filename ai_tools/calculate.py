import sympy

CALCULATE_META = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description":(
            "Evaluate a mathematical expression and return the exact result. "
            "Use this for arithmetic, numerical calculations, and unit-free "
            "mathematical expressions. Use this tool instead of calculating "
            "results mentally. Prefer standard mathematical functions when "
            "available rather than recreating them from more primitive operations. "
            "For example, use sqrt(x) rather than x**0.5 and hypot(a, b) rather "
            "than sqrt(a**2 + b**2). Mathematical functions such as sin, sqrt, "
            "cbrt, and hypot use standard function notation with values or inner "
            "expressions inside parentheses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate.",
                },
            },
            "required": ["expression"],
        },
    },
}

def calculate(expression: str) -> str:
    try:
        result = sympy.sympify(expression)
        return str(result)
    except (sympy.SympifyError, TypeError, ValueError) as exc:
        print(f"calculate({expression!r}) failed: {exc}")
        return "Could not evaluate expression."