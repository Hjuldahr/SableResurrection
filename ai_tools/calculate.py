import sympy

def calculate(expression: str) -> str:
    try:
        result = sympy.sympify(expression)
        return str(result)
    except (sympy.SympifyError, TypeError, ValueError) as exc:
        print(f"calculate({expression!r}) failed: {exc}")
        return "Could not evaluate expression."