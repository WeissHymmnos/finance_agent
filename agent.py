import os
import requests
import asyncio
import nest_asyncio
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field
from sec_edgar_downloader import Downloader
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from ddgs import DDGS
from llama_index.llms.openai_like import OpenAILike
# LlamaIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
# 动态代码解释器沙盒
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec
from llama_index.core.agent import ReActAgent
# LlamaIndex引擎
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

nest_asyncio.apply()

# 提取 API Keys
MOONSHOT_API_KEY = os.getenv("LLM_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# 分析目标
TARGET_TICKER = "NVDA"

# 1. 定义LLM
# 1. 定义 LlamaIndex 专属的“正规军”大脑
# 注意：这里我们使用刚才 as 出来的 LlamaOpenAI，且参数名必须是 api_base
# 1. 定义 LlamaIndex 专属的“第三方兼容”大脑
llama_llm = OpenAILike(
    model="moonshot-v1-32k",
    api_key=MOONSHOT_API_KEY,
    api_base="https://api.moonshot.cn/v1", 
    is_chat_model=True,
    context_window=32000,
    temperature=0,
    additional_kwargs={"extra_body": {"chat_template_kwargs": {"thinking": False}}}
)
# 2. 全局绑定：让 RAG 引擎默认使用这个纯正的 LlamaIndex 大脑
Settings.llm = llama_llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")# ==========================================

# 2. 签署数据契约 (Pydantic 结构化模型)
# ==========================================
class CompanyAnalysis(BaseModel):
    ticker: str = Field(..., description="股票代码")
    dynamic_quant_metrics: str = Field(..., description="代码解释器动态计算出的量价指标分析")
    sec_risk_factors: str = Field(..., description="本地 RAG 从 SEC 10-K 中提取的核心风险")

class FinalComparativeReport(BaseModel):
    tesla_analysis: CompanyAnalysis
    ford_analysis: CompanyAnalysis
    investment_verdict: str = Field(..., description="基于量价与基本面的中英双语对冲/投资建议")

# ==========================================
# 3. 定义流水线事件 (Event-Driven Flow)
# ==========================================
class QuantDataEvent(Event):
    """事件节点：动态量化数据计算完毕"""
    ticker: str
    quant_result: str

# ==========================================
# 4. 构建单股深度分析流水线 (Workflow)
# ==========================================
class SingleStockQuantWorkflow(Workflow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 1. 实例化代码沙盒工具
        code_tool = CodeInterpreterToolSpec()
        
        self.code_agent = ReActAgent(
            tools=code_tool.to_tool_list(), 
            llm=llama_llm,
            verbose=True,
            max_iterations=10
        )       

        # 初始化 SEC 下载器
        self.dl = Downloader("QuantLab", "research@quant.com")

    @step
    async def step_1_dynamic_quant(self, ev: StartEvent) -> QuantDataEvent:
        """步骤 1：大模型自主写 Python 代码获取并计算数据"""
        ticker = ev.ticker
        print(f"[{ticker}] 启动代码沙盒：正在自主编写并执行代码计算波动率与回撤...")
        
        prompt = f"""
        Write and execute Python code to:
        1. Fetch the last 6 months of historical close prices for {ticker} using yfinance.
        2. Calculate the Annualized Volatility and Maximum Drawdown.
        3. Print ONLY the final numerical results clearly.
        """

        # 爆改这里：把 input 换成官方硬性要求的 user_msg
        response = await self.code_agent.run(user_msg=prompt)
        
        # 稳妥提取返回文本
        final_answer = str(response)
        
        return QuantDataEvent(ticker=ticker, quant_result=final_answer)

    @step
    async def step_2_local_sec_rag(self, ev: QuantDataEvent) -> StopEvent:
        """步骤 2：利用本地 RAG 提取财报风险，并封装为 Pydantic 对象返回"""
        ticker = ev.ticker
        print(f"[{ticker}] 启动本地 RAG：正在拉取并研读 SEC 10-K 财报...")
        
        target_path = f"sec-edgar-filings/{ticker}/10-K"
        if not os.path.exists(target_path):
            self.dl.get("10-K", ticker, limit=1, download_details=True)
            
        # 安全读取，过滤非文本文件
        doc = SimpleDirectoryReader(
            input_dir=target_path, 
            recursive=True, 
            required_exts=[".txt", ".htm", ".html"]
        ).load_data()
        
        index = VectorStoreIndex.from_documents(doc)
        rag_engine = index.as_query_engine(similarity_top_k=2)
        
        risk_response = rag_engine.query(
            f"What are the specific supply chain constraints and margin risks highlighted by management for {ticker}?"
        )
        
        # 严格按照 Pydantic 格式打包这只股票的所有数据
        analysis = CompanyAnalysis(
            ticker=ticker,
            dynamic_quant_metrics=ev.quant_result,
            sec_risk_factors=str(risk_response)
        )
        print(f"[{ticker}]  单股分析流转完毕！")
        return StopEvent(result=analysis)

# ==========================================
# 5. 司令部：并发调度与终极对决
# ==========================================
async def main():
    print("========================================================")
    print(" 启动：TSLA vs F 动态量价与基本面对比")
    print("========================================================\n")
    
    # 1. 实例化工作流引擎
    workflow = SingleStockQuantWorkflow(timeout=600)
    
    # 2. 并发拉起两只股票的分析流
    task_tsla = workflow.run(ticker="TSLA")
    task_f = workflow.run(ticker="F")
    
    # 等待双线任务全部完成
    tsla_result, f_result = await asyncio.gather(task_tsla, task_f)
    
    print("\n 双方数据收集完毕，主脑正在进行终极结构化对比，请稍候...")
    
    # 3. 强制结构化输出对比报告
    prompt = f"""
    You are a Senior Quantitative Analyst. Compare TSLA and F based on the following raw agent data:
    
    TSLA Data: {tsla_result.model_dump_json()}
    Ford (F) Data: {f_result.model_dump_json()}
    
    Output your comprehensive comparison STRICTLY conforming to the FinalComparativeReport JSON schema.
    Ensure the 'investment_verdict' section is detailed and written in Professional Chinese.
    """
    
    final_report = await llama_llm.astructured_predict(
        FinalComparativeReport,
        prompt=prompt
    )
    
    print("\n 终极量化研报 (标准 JSON 格式交付)：")
    print(final_report.model_dump_json(indent=4))

if __name__ == "__main__":
    asyncio.run(main())
