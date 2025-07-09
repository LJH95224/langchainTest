from langchain_core.output_parsers import BaseOutputParser


class CommaSeparatedListOutputParser(BaseOutputParser):
    """将LLM调用的输出解析为逗号分隔的列表."""

    def parse(self, text: str):
        """解析LLM调用的输出."""
        # text.strip()：去除字符串首尾的空白符。
        # .split(", ")：按 ", " 分割字符串，返回列表。
        return text.strip().split(", ")


# split 按", "分割，包含空格
response = CommaSeparatedListOutputParser().parse("hi, bye")
print(response)  # ['hi', 'bye']

# split 按", "分割，不包含空格
response2 = CommaSeparatedListOutputParser().parse("hi,bye")
print(response2)  # ['hi,bye']
