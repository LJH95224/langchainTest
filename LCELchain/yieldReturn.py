# 使用 return
def get_squares_return(n):
    """返回包含 0 到 n - 1 的平方的列表"""
    result = []
    for i in range(n):
        result.append(i * i)
    return  result

# 使用return 函数
squares = get_squares_return(5)
print("使用 'return' 的结果", squares) # 一次性获取所有的结果
print("类型： ", type(squares))

for num in squares:
    print(num)

print("再次遍历 return")
for num in squares:
    print(num)


# 使用 yield
def get_squares_yield(n):
    """返回包含 0 到 n - 1 的平方的列表"""
    for i in range(n):
        yield i * i

# 使用 yield 函数
squares_gen = get_squares_yield(5)
print("使用 'yield' 的结果", squares_gen)
print("类型： ", type(squares_gen))

# 遍历生成器
for num in squares_gen:
    print(num)


# 再次遍历
print("再次遍历 yield")
for num in squares_gen:
    print(num) # 不会输出任何内容，因为生成器已经被消耗完毕了