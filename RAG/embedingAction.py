# 向量数据库操作

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(verbose=True, override=True)


siliconflow_api_key = os.environ["SILLICONFLOW_API_KEY"]
siliconflow_base_url = os.environ["SILLICONFLOW_API_BASE"]

embedding_model = OpenAIEmbeddings(
    model="netease-youdao/bce-embedding-base_v1",
    openai_api_key=siliconflow_api_key,
    openai_api_base=siliconflow_base_url,
)

# 引入内存向量数据库，它将向量暂存在内存中，并使用字典以及numpy计算搜索的余弦相似度
from  langchain_core.vectorstores import InMemoryVectorStore
vectorstore = InMemoryVectorStore(embedding_model)

from langchain_core.documents import Document

document_1 = Document(
    page_content="今天在抖音学会了一个新菜:锅巴土豆泥!看起来简单，实际炸了厨房。",
    metadata={"source": "社交媒体"}
)

document_2 = Document(
    page_content="小区遛狗大爷今日播报:广场舞大妈占领健身区，遛狗群众纷纷撤退。现场气氛诡异，BGM已循环播放《最炫民族风》两小时。",
    metadata={"source": "社区新闻"}
)



documents = [document_1, document_2]

embedding_Index = vectorstore.add_documents(documents)
# print(f"embedding_Index: {embedding_Index}\n")


# 可以为添加的文档增加ID索引，便于后面管理
# embedding_Index2 = vectorstore.add_documents(documents=documents, ids=["doc1", "doc2"])
# print(f"embedding_Index2: {embedding_Index2}\n")

more_docs = [
    Document(page_content="今天猫咪不吃饭了，只想睡觉。", metadata={"source": "宠物日志"}),
    Document(page_content="关于OpenAI最新模型GPT-5的性能提升分析", metadata={"source": "科技资讯"}),
    Document(page_content="今天吃了好吃的冒菜，香得很！", metadata={"source": "美食日记"}),
    Document(page_content="我家狗子早上把我的鞋藏起来了，找了一个小时才发现它叼去阳台晒太阳了。",
             metadata={"source": "宠物趣事"}),
    Document(page_content="AI绘画越来越卷了，今天试了下Midjourney V6，效果比我画得还好。", metadata={"source": "AI体验"}),
    Document(page_content="最近高温预警不断，出门五分钟，流汗两小时。", metadata={"source": "生活观察"}),
    Document(page_content="深夜厨房日记：第一次做拉面，面没拉好，锅却干了。", metadata={"source": "深夜厨房"}),
    Document(page_content="孩子写作业时突然问我地球是几维的，我差点没答上来。", metadata={"source": "家长日常"}),
    Document(page_content="今天看到一只胖橘猫在马路边拦车，像个交通警。", metadata={"source": "城市随拍"}),
    Document(page_content="听说GPT能写代码，我让它写了个贪吃蛇游戏，真的能跑！", metadata={"source": "技术实验"}),
    Document(page_content="最近追剧《繁花》，张颂文太有味道了，那个眼神简直绝了。", metadata={"source": "追剧笔记"}),
    Document(page_content="今早地铁上人超级多，有人拿着蒜香肠，整节车厢都饿了。", metadata={"source": "通勤趣事"}),
    Document(page_content="三亚的海真美，椰风、沙滩、还有冰镇椰汁，绝了！", metadata={"source": "旅游分享"}),

    Document(page_content="猫真的很会挑地方躺，它今天躺在键盘上不让我写代码。", metadata={"source": "程序员日常"}),
    Document(page_content="用了一个月Obsidian做笔记，感觉比Notion更适合写技术文档。", metadata={"source": "工具体验"}),
    Document(page_content="今天听了张学友的演唱会直播，还是那么稳，经典不老。", metadata={"source": "音乐分享"}),
    Document(page_content="AI换脸视频最近太火了，但也有点恐怖，真假难辨。", metadata={"source": "科技观察"}),
    Document(page_content="健身房遇到一个动作全错的猛男，教练在旁边都不敢说。", metadata={"source": "健身趣事"}),
    Document(page_content="今天面试被问到了线程池底层实现，还好前几天刚复习过。", metadata={"source": "求职经历"}),
    Document(page_content="最近有点抑郁，晚上总是睡不着，听听lofi稍微缓解一些。", metadata={"source": "情绪日记"}),
    Document(page_content="试了ChatGPT翻译一段德语，比我自己翻得顺多了。", metadata={"source": "AI助手"}),
    Document(page_content="隔壁小孩弹钢琴已经弹了一周《小星星》，求放过。", metadata={"source": "邻居日常"}),
    Document(page_content="想给阳台种点小番茄，但担心我连仙人掌都养不活。", metadata={"source": "阳台计划"})
]

vectorstore.add_documents(documents=more_docs)



# 向量库删除
# embedding_del = vectorstore.delete(ids=["doc1"])
# print(f"embedding_del: {embedding_del}\n")


# 向量库近似搜索
query = "锅巴土豆泥"
docs = vectorstore.similarity_search(query)
# print(f"docs: {docs}\n ")
print(f"docs[0].page_content: {docs[0].page_content}")


# 使用“向量”查相似“向量”的方式来进行搜索
query2 = "遛狗"
embedding_vector = embedding_model.embed_query(query2)
docs2 = vectorstore.similarity_search_by_vector(embedding_vector)
# print(f"docs2: {docs2}\n ")
print(f"docs2[0].page_content: {docs2[0].page_content}")

"""
为什么用 "锅巴土豆泥" 和 "遛狗" 两个完全不同的查询，结果 docs[0] 和 docs2[0] 是一样的？
📌 具体原因详解：
	1.	similarity_search(query) 会将你的 query 文本转为 embedding 向量，然后和库中所有文档的向量进行余弦相似度计算。
	2.	返回的是 最相似的 top-k（默认是 k=4，如果没指定），而不是“相似度超过某个阈值”的过滤式搜索。
	3.	因为你只插入了 2 个文档，无论你查啥，它总会返回一个（top-1），哪怕这个“相似度其实很低”。
	4.	所以你看到 docs[0] 和 docs2[0] 是一样的，是因为它都返回了最接近但不代表相关的那条记录。
"""