# 使用 RunnablePassthrough 来传递值
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 创建一个可并行运行的处理流程
runnable = RunnableParallel(
    passed = RunnablePassthrough(),  # 第一个处理器： 直接传递输入，不做修改
    modified = lambda x: x["num"] + 1, # 第二个处理器： 取出输入中的num字段，并加1
)

# 执行这个处理流程, 传入一个包含 num 字段的字典
response = runnable.invoke({"num": 1})
print(response)
# {'passed': {'num': 1}, 'modified': 2}