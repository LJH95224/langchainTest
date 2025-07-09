# deepseek 模型区别

| 对比项               | DeepSeek Chat            | DeepSeek Reasoner            |
|-------------------|--------------------------|------------------------------|
| **模型定位**          | 通用对话与指令执行                | 强逻辑推理与多步任务求解                 |
| **主要用途**          | 问答、内容生成、翻译、总结、指令跟随       | 数学推理、因果分析、算法题解、多轮推理任务        |
| **训练侧重**          | 自然语言交互、RLHF、SFT          | 思维链（CoT）、工具使用、逻辑训练、长链推理      |
| **是否支持工具调用**      | ❌ 不支持或弱集成                | ✅ 支持工具/函数调用机制                |
| **推理能力**          | ⭐⭐⭐（一般）                  | ⭐⭐⭐⭐⭐（强）                     |
| **典型场景**          | 日常 AI 助手、总结文案、生成文本       | 解数学题、代码推理、多轮对话中保持逻辑一致性       |
| **适合构建 AI Agent** | ✅ 可以用于基础任务               | ✅✅ 更适合复杂 Agent 推理任务          |
| **示例对话**          | “请写一段春天的作文” → 输出结构清晰优美文章 | “3个贴错标签的箱子如何判断” → 输出详细逻辑步骤推理 |

> ✅ 总结：**DeepSeek Chat** 擅长自然语言对话与通用任务，**Reasoner** 擅长逻辑严谨的推理与复杂问题求解。
> 相比较而言，Reasoner 比 Chat 更加消耗token， 返回结果中 Reasoner 包含推理逻辑（additional_kwargs['reasoning_content']
> ），而Chat里面没有

# LangChain 标准参数、事件、与输入输出

## 标准事件

- **invoke**: 模型主要调用方法，输入list，输出list
- **stream**: 流式输出方法
- **batch**: 批量模型请求方法
- **bind_tools**: 在模型执行的时候绑定工具
- **with_structured_output**: 基于invoke的结构化输出

## 其他有用的事件

- **ainvoke**: 异步调用模型的方法
- **astream**: 异步流式输出
- **abatch**: 异步的批量处理
- **astream_events**: 异步流事件
- **with_retry**：调用失败时重试
- **with_fallback**: 失败恢复事件
- **configurble_fields**: 模型运行时的运行参数

# python 虚拟环境

## 设置虚拟环境
```bash
python3 -m venv .venv
```

## 导出依赖
```bash
pip3 freeze > requirements.txt
```

### 退出虚拟环境
```bash
deactivate
```

### 安装 requirements.txt 中的依赖包
```bash
pip3 install -r requirements.txt
```

### 安装pipreqs  
> pipreqs 是一个非常实用的工具，它可以根据你的项目代码，自动生成一个 requirements.txt 文件，只包含你实际使用到的 Python 包（而不是像 pip freeze 一样列出整个虚拟环境中的所有包）。

#### 安装
```bash
pip3 install pipreqs
```

#### 基本用法
> 在项目根目录下运行
```bash
pipreqs .
```

#### 常用参数

| 参数    | 用说明 |
|-------|--------------------------|
| --force | 强制覆盖已存在的 requirements.txt 文件 |
| --encoding utf-8 | 指定文件编码，避免编码错误 |
| --ignore <dir> | 忽略某些目录（如 venv, tests）|
| --savepath <path> | 指定 requirements.txt 保存路径 |
| --proxy <proxy> | 使用代理连接 PyPI 以查询依赖版本

示例：
```bash
pipreqs . --force --encoding=utf-8 --ignore tests,venv
```

#### pipreqs vs pip freeze 对比

| 比较项   | pipreqs        | pip freeze    | 
|-------|----------------|---------------|
| 作用    | 只列出项目中实际用到的库   | 列出虚拟环境中安装的所有库 | 
| 用途    | 更适合开源项目或最小依赖清单 | 更适合部署时完整还原环境  | 
| 是否含未使用包 | ❌ 不含           | ✅ 全部包含        |

#### 注意事项
> 1. pipreqs 依赖源文件中的 import 语句，不会识别动态导入。 
> 2. 如果你用的是模块别名（如 import numpy as np），它也能识别。