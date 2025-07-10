from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.utilities import PythonREPL
# from langgraph.prebuilt import ToolNode
from typing import List, Annotated
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools import StructuredTool, ToolException
try:
    from api.qdrant_cloud_ops import connect_to_qdrant
except ImportError:
    from qdrant_cloud_ops import connect_to_qdrant

client = connect_to_qdrant()
COLLECTION_NAME = 'aireas-cloud'
EMBEDDING_MODEL= GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')

class QdrantRetriever(BaseRetriever):
    client_: QdrantClient
    embedding_model_: GoogleGenerativeAIEmbeddings
    collection_name_: str 
    with_payload_: bool 
    limit_: int  

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # Generate query embeddings
        query_embeddings = self.embedding_model_.embed_query(query)

        # Perform a search in Qdrant using the client
        search_result = self.client_.query_points(
            collection_name=self.collection_name_,
            query=query_embeddings,
            with_payload=self.with_payload_,
            limit=self.limit_,
        )

        # Extract documents from search results
        documents = []
        if hasattr(search_result, 'points'):
            for point in search_result.points:
                document = Document(
                    metadata={"pdf_id": point.payload.get("pdf_id", ""), "score": point.score},
                    page_content=point.payload.get("text", ""),
                    
                )
                documents.append(document)
        return documents


# Instantiate QdrantRetriever with required parameters
Qretriever = QdrantRetriever(
    client_=client,
    collection_name_=COLLECTION_NAME,
    embedding_model_=EMBEDDING_MODEL,
    limit_=2,
    with_payload_=True
)

# Load other tools
arxiv_search_tool = load_tools(["arxiv"])[0]


tavily_search_tool = TavilySearchResults(max_results=3)

# Web scraper class
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )

web_scraper_tool = StructuredTool.from_function(
    func=scrape_webpages,
    handle_tool_error=True
)

# Python REPL tool
repl = PythonREPL()

def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

repl_tool = StructuredTool.from_function(
    func=python_repl,
)


# research supervisor prompt
research_supervisor_prompt = (
    "You are a supervisor tasked with managing a research team with each worker utilizing a specific tool: TavilySearch, WebScraper, PythonReplt, ArxivSearch, QdrantRetriver "
    "Given the following user task, develop a comprehensive step-by-step plan to address the research question. "
    "Each plan should specify which external tool will be utilized, along with the necessary input required to gather evidence. "
    "Evidence can be stored in variables labeled as #E, which can be referenced by subsequent plans. "
    "Structure your plans as follows: (Plan, #E1, Plan, #E2, Plan, ...).\n\n"
    
    "Tools descriptions:\n"
    "1. Tavily Search Tool: Ideal for retrieving concise answers or insights from various online sources.\n"
    "2. Web Scraper Tool: Suitable for extracting detailed information from specific web pages.\n"
    "3. Python REPL Tool: Executes Python code for computational tasks or data analysis.\n"
    "4. Arxiv Search Tool: Searches for relevant academic papers and articles on arXiv.\n"
    "5. Qdrant Retriever Tool: Performs similarity searches on the Qdrant vector database to find the most relevant documents or answers related to the given query.\n\n"

    "Example Task 1:\n"
    "Task: What are the key differences between BERT and GPT models in terms of architecture and performance?\n"
    "Example Plan:\n"
    "- Plan: Conduct a similarity search on the Qdrant vector database using the query 'BERT vs GPT architecture and performance' to retrieve relevant documents. #E1 = QdrantRetriever['BERT vs GPT architecture and performance']\n"
    "- Plan: Use the Arxiv Search Tool to find recent papers comparing BERT and GPT models, focusing on their architectural differences and performance metrics. #E2 = ArxivSearchTool['BERT vs GPT performance comparison']\n"
    "- Plan: Summarize the findings from #E1 and #E2 using the Python REPL Tool to create a comparative analysis. #E3 = PythonREPL['Summarize findings from #E1 and #E2']\n\n"

    "Example Task 2:\n"
    "Task: Can you provide a brief overview of the most cited papers in reinforcement learning from the last year?\n"
    "Example Plan:\n"
    "- Plan: Search for articles and summaries on Tavily regarding the most cited reinforcement learning papers published in the last year. #E1 = TavilySearchTool['most cited reinforcement learning papers 2023']\n"
    "- Plan: Retrieve the top reinforcement learning papers from arXiv to gather detailed information. #E2 = ArxivSearchTool['top cited reinforcement learning papers 2023']\n\n"

    "Example Task 3:\n"
    "Task: What are the latest techniques for data augmentation in deep learning, and how can I implement one using Python?\n"
    "Example Plan:\n"
    "- Plan: Use the Web Scraper Tool to extract detailed techniques for data augmentation from a specific webpage that lists the latest methods. #E1 = WebScraperTool['URL of a relevant webpage on data augmentation']\n"
    "- Plan: Implement one of the techniques using Python. #E2 = PythonREPL['Implement a data augmentation technique in Python']\n\n"

    "Begin!\n"
    "Describe your plans with rich details. Each plan should be followed by only one #E.\n"
    
    "Task: {task}"
)