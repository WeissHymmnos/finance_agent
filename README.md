# Finance Agent

A quantitative and qualitative financial analysis agent that uses Large Language Models to analyze stock performance and SEC filings.

## Overview

This project implements an automated workflow for stock analysis. It combines technical indicators (Volatility, Maximum Drawdown) fetched via yfinance with fundamental risk analysis extracted from SEC 10-K filings using LlamaIndex and RAG. The system provides a web interface for users to input ticker symbols and receive a comprehensive comparative investment report.

## Key Features

- Automated Quantitative Analysis: Calculates annualized volatility and maximum drawdown using an LLM-driven code interpreter.
- SEC Filing RAG: Downloads, parses, and indexes SEC 10-K filings to identify supply chain constraints and margin risks.
- Comparative Reporting: Generates structured investment verdicts comparing multiple stocks.
- Web Interface: A Flask-based web dashboard for easy interaction.
- Workflow Management: Utilizes LlamaIndex Workflows for multi-step, asynchronous task execution.

## Project Structure

- agent.py: The main application file containing the Flask server and LlamaIndex workflows.
- environment.yml: Conda environment configuration with all necessary dependencies.
- templates/index.html: The front-end user interface.
- sec-edgar-filings/: Local directory where downloaded SEC filings are stored.

## Prerequisites

- Python 3.12
- Conda (recommended for environment management)
- Moonshot API Key (configured as LLM_KEY environment variable)

## Installation

1. Clone the repository to your local machine.
2. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```bash
   conda activate finance_agent
   ```
4. Set up your environment variables:
   ```bash
   export LLM_KEY="your_moonshot_api_key"
   ```

## Usage

1. Start the Flask server:
   ```bash
   python agent.py
   ```
2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
3. Enter ticker symbols (e.g., AAPL, TSLA) and the desired analysis period to generate a report.

## Technical Details

- LLM Provider: Moonshot (via OpenAILike interface).
- Embedding Model: BAAI/bge-small-en-v1.5.
- Data Sources: Yahoo Finance (yfinance) and SEC EDGAR.
- Frameworks: Flask, LlamaIndex, LangChain.
