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
from llama_index.core import VectorStoreIndex, PromptTemplate, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
nest_asyncio.apply()

MOONSHOT_API_KEY = os.getenv("LLM_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

llama_llm = OpenAILike(
    model="moonshot-v1-32k",
    api_key=MOONSHOT_API_KEY,
    api_base="https://api.moonshot.cn/v1", 
    is_chat_model=True,
    context_window=32000,
    temperature=0,
    additional_kwargs={"extra_body": {"chat_template_kwargs": {"thinking": False}}}
)

Settings.llm = llama_llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

class CompanyAnalysis(BaseModel):
    ticker: str = Field(..., description="股票代码")
    dynamic_quant_metrics: str = Field(..., description="量化指标文字描述")
    volatility: float = Field(..., description="年化波动率(数值，例如0.25)")
    max_drawdown: float = Field(..., description="最大回撤(数值，例如-0.15)")
    sec_risk_factors: str = Field(..., description="SEC 财报核心风险")

class FinalComparativeReport(BaseModel):
    analyses: dict[str, CompanyAnalysis] = Field(..., description="各股票分析结果映射")
    investment_verdict: str = Field(..., description="投资建议(中英双语)")

class QuantDataEvent(Event):
    ticker: str
    quant_result: str

class SingleStockQuantWorkflow(Workflow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dl = Downloader("QuantLab", "research@quant.com")

    @step
    async def step_1_dynamic_quant(self, ev: StartEvent) -> QuantDataEvent:
        """执行 Python 代码计算量化指标"""
        ticker, months = ev.ticker, ev.months
        print(f"[{ticker}] 正在执行代码沙盒计算指标...")
        
        local_code_agent = ReActAgent(
            tools=CodeInterpreterToolSpec().to_tool_list(), 
            llm=llama_llm,
            verbose=False,
            max_iterations=10
        )
        
        prompt = f"""
        Write and execute Python code to:
        1. Fetch the last {months} months of historical close prices for {ticker} using yfinance.
        2. Calculate the Annualized Volatility: (standard deviation of daily returns) * sqrt(252).
        3. Calculate the Maximum Drawdown (MDD): (Current_Price - Running_Peak) / Running_Peak.
        4. In your FINAL ANSWER, you MUST include these exact markers with numerical values:
           VOLATILITY: <value>
           MAX_DRAWDOWN: <value>
        
        Example: VOLATILITY: 0.253, MAX_DRAWDOWN: -0.124
        """
        response = await local_code_agent.run(user_msg=prompt)
        return QuantDataEvent(ticker=ticker, quant_result=str(response))

    @step
    async def step_2_local_sec_rag(self, ev: QuantDataEvent) -> StopEvent:
        """提取 SEC 财报风险"""
        ticker = ev.ticker
        print(f"[{ticker}] 正在研读 SEC 10-K 财报...")
        
        target_path = f"sec-edgar-filings/{ticker}/10-K"
        if not os.path.exists(target_path):
            self.dl.get("10-K", ticker, limit=1, download_details=True)
            
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
        
        analysis = CompanyAnalysis(
            ticker=ticker,
            dynamic_quant_metrics=ev.quant_result,
            volatility=0.0,
            max_drawdown=0.0,
            sec_risk_factors=str(risk_response)
        )
        return StopEvent(result=analysis)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
async def analyze():
    data = request.json
    tickers = data.get('tickers', ['TSLA', 'F'])
    months = data.get('months', 6)
    
    workflow = SingleStockQuantWorkflow(timeout=600)
    tasks = [workflow.run(ticker=t, months=months) for t in tickers]
    results = await asyncio.gather(*tasks)
    
    # 构造更清晰的上下文
    results_context = ""
    for r in results:
        results_context += f"--- TICKER: {r.ticker} ---\n"
        results_context += f"QUANT_DATA: {r.dynamic_quant_metrics}\n"
        results_context += f"SEC_RISKS: {r.sec_risk_factors}\n\n"

    prompt = f"""
    You are a Senior Quantitative Analyst. Analyze the following data for {', '.join(tickers)}:
    
    {results_context}
    
    Task:
    1. Extract numerical values for 'volatility' and 'max_drawdown' from each 'QUANT_DATA' block.
    2. Ensure 'max_drawdown' is represented as a negative float (e.g., -0.15).
    3. If a value is missing or unclear, use 0.0.
    4. Provide a comparative investment verdict in Professional Chinese.
    
    Output STRICTLY in JSON format following the FinalComparativeReport schema.
    Key in 'analyses' must be the ticker symbol.
    """
    
    final_report = await llama_llm.astructured_predict(
        FinalComparativeReport,
        prompt=PromptTemplate(prompt)
    )
    return jsonify(final_report.model_dump())

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
