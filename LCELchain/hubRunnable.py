from langchain.runnables.hub import HubRunnable
from langchain_core.runnables import ConfigurableField
import os
# 如果你不打算使用 LangSmith，可以在代码顶部添加
os.environ["LANGCHAIN_TRACING_V2"] = "false"


# 加载一个 Hub 上的 prompt 配置
prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
    owner_repo_commit = ConfigurableField(
        id="hub_commit",
        name="Hub Commit",
        description="The Hub commit to pull from",
    )
)

# 调用该 prompt（传入 context 和 question）
response = prompt.invoke({"question": "foo", "context": "bar"})
print( response)

response2 = prompt.with_config(configurable={"hub_commit": "rlm/rag-prompt-llama"}).invoke({"question": "foo", "context": "bar"})
print( response2)