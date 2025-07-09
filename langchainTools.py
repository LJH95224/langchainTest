from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """"将两数字相乘"""
    return a * b

# print(multiply(2, 3))
print(multiply.name)
print(multiply.description)
print(multiply.args)