from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import List, Optional, Union, TypedDict, Literal, Dict, Literal, Annotated
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages
try:
    from api.token_counter import tiktoken_counter
except ImportError:
    from token_counter import tiktoken_counter

from langchain.schema import SystemMessage, HumanMessage

trimmer_first = trim_messages(
    max_tokens=1500,
    strategy="first",
    token_counter=tiktoken_counter,
    # include_system=False,
    allow_partial=True,
)

class DecomposedQuestion(TypedDict):
    """
    Represents decomposed sub-questions.
    """
    sub_questions: Annotated[
        Optional[List[str]], 
        "A list of sub-questions derived from the original question."
    ]

class RephrasedQuestion(TypedDict):
    """
    Represents rephrased question.
    """
    rephrased_question: str

class DecompositionCheck(TypedDict):
    """
    For evaluating whether a question needs to be decomposed into simple, isolated, answerable sub-questions.
    """
    needs_decomposition: Annotated[Literal["Decompose", "Rephrase"], "Evaluation of whether the question needs to be decomposed into simpler sub-questions."
    ]

class DecideMemorise(TypedDict):
    action: Literal['memorize', 'none']
    content: str


def dec_parser(task_list: List[str]) -> str:
    """
    Parses a list of tasks, removing any empty or whitespace-only strings,
    and returns a single string with tasks separated by commas.

    Parameters:
    - task_list (list): The list of task strings to be parsed.

    Returns:
    - str: A non-empty, comma-separated string of tasks.
    """
    parsed_task = ', '.join(filter(lambda x: x.strip() != '', task_list))
    return parsed_task


def decomposition_chain(llm):
    """
    Creates a chain for decomposing questions into sub-questions or rephrasing the question.
    
    Parameters:
    - llm: The language model instance.

    Returns:
    - callable: A callable chain that performs question decomposition.
    """
    template = """You are an assistant that decomposes complex questions into simpler, isolated sub-questions.

    Your task is to break down the given question into the smallest possible set of essential sub-problems or sub-questions, with a maximum limit of 3 sub-questions.
    ### Important and strict note: Break down the question into sub-questions **only if it improves clarity or retrieval efficiency**. If the question can be directly answered or requires only one or two sub-questions, do not force it into three sub-questions.

    Each sub-question should:
    1. Be independently answerable.
    2. Be necessary to address the original question.
    3. Be formulated in a way that enhances the efficiency of the retrieval process.

    Respond only with the sub-questions, separated by a newline.

    Question: {question}
    """

    structured_output_llm = llm.with_structured_output(DecomposedQuestion)

    decomposition_prompt = ChatPromptTemplate.from_template(template)

    generate_queries_decomposition = (
        decomposition_prompt 
        | structured_output_llm 
        | (lambda x: dec_parser(x["sub_questions"]).lower())
    )
    return generate_queries_decomposition


def requires_decomposition(llm):
    template = """You are an assistant that evaluates whether a given question should be decomposed into simpler, isolated sub-questions.

    Your task is to determine if the given question needs to be broken down. 
    - Answer "Decompose" if the question is complex, multi-faceted, or requires multiple retrieval tasks to answer.
    - Answer "Rephrase" if the question is straightforward or can be answered directly without decomposition.

    Question: {question}
    """

    structured_output_llm = llm.with_structured_output(DecompositionCheck)

    decomposition_prompt = ChatPromptTemplate.from_template(template)

    evaluate_decomposition = decomposition_prompt | structured_output_llm | (lambda x: x["needs_decomposition"])

    return evaluate_decomposition


def rephrase_chain(llm):
    template = """You are an assistant tasked with rephrasing complex or unclear questions into simpler, clearer, and more direct versions while preserving the original meaning.
    
    Your goal is to improve the clarity and search efficiency of the input question for document retrieval systems like web search, Arxiv, or other knowledge databases. You should focus on:
    
    1. Making the question more concise and to the point.
    2. Retaining key domain-specific terms that are necessary for accurate retrieval (e.g., technical terms, scientific terms).
    3. Simplifying sentence structures, but ensuring that the question targets the core aspect of the query.
    4. Replacing general or vague phrasing with more specific, searchable keywords.
    5. Making sure that the question is actionable, targeting a clear and direct search outcome.

    If the question includes technical or scientific terms, leave them as they are but ensure they are used effectively for search efficiency. Do not introduce ambiguity.

    The rephrased question should be well-suited for searching in databases like Arxiv, Google Scholar, or other document retrieval systems.

    Question: {question}
    """

    # Ensure you have the right way to structure output based on your framework
    structured_output_llm = llm.with_structured_output(RephrasedQuestion)  # Replace RephrasedQuestion if needed
    
    # Define your prompt template properly
    rephrasing_prompt = ChatPromptTemplate.from_template(template)
    
    # Rephrase the question and process the output
    rephrase_chain = (
        rephrasing_prompt 
        | structured_output_llm 
        | (lambda x: x["rephrased_question"].lower())  # Adjust based on the structure of output
    )
    
    return rephrase_chain



def get_plan_chain(llm):
    few_shot_rewoo = """
    You are a planner agent.
    Supervisor will prvide you with a task and you will develop a sequential plan that specifies exactly which agents to use to retrieve the necessary evidence. 
    IMPORTANT: ALWAYS TAKE MINIMUM NUMBER OF PLAN STEPS TO ACCOMPLISH THE GIVEN TASK.
    When addressing the query follow the below format to create a step by step plan.

    Format:
    Plan: [Provide a concise description of the intended action, including any specific sources, search queries, or steps that must be followed. Reference any evidence needed from previous steps.]
    #E[number] = [Agent[Specific Query/Input, including any references to previous #E results if applicable]]

    Instructions for Plan Creation:
    - Use the minimum number of plans necessary to provide an accurate and relevant answer.
    - Each plan should be followed by only one #E, with clear sequential ordering for reference by subsequent steps. (STRICTLY FOLLOW THE GIVEN FORMAT)
    - Create a complete plan that addresses all subquestions as a whole, rather than developing individual plans for each question separately.
    - PLEASE TAKE NOTE - FOR MOST CASES YOU WILL NOT BE REQUIRED TO USE ALL THE AGENTS FOR THE GIVEN PLAN. ADJUST ACCORDING TO COMPLEXITY OF THE USER QUESTION.
    - Use the 'Coder' agent only when Python code is required for calculations, visualizations, or when explicitly asked to write code. Ensure you fully understand the question before resorting to the 'Coder' agent.

    Agents Available:
    - RagSearcher[input]: Retrieves relevant documents or research papers using a vector database (qdrant_retriever_tool). Ideal for tasks involving PDFs or embedded topics.
    - Searcher[input]: Performs web searches with tavily_search_tool for general info or academic papers from online sources.
    - ArXivSearcher[input]: Searches for academic papers on ArXiv using arxiv_search_tool.
    - Coder[input]: Executes Python code via repl_tool for programming, data analysis, and visualizations.
    - ChatBot[input]: Generates natural language responses based on input or gathered evidence for conversational tasks.

    Instructions on Considering Conversation History:
    - You will also be able to read last few messages of the given conversation (whichever could be fitted in your context window). Incorporate relevant information or context from past messages when creating plans.
    - Summarize past messages, if necessary, to create precise and well-informed agent queries.


    Advice:
    - Ensure that each agent query reflects the task at hand and leverages any relevant conversation history or past messages to optimize results.
    
    Example 1:
    Task: Summarize recent advancements in Video Transformers for action recognition tasks.
    Plan: Search for recent publications on Arxiv that discuss Video Transformers for action recognition using the Searcher agent.
    #E1 = Searcher[Video Transformers action recognition]
    Plan: Retrieve related research papers on Video Transformers stored in the vector database using the RagSearcher agent.
    #E2 = RagSearcher[Video Transformers action recognition]
    Plan: Use the retrieved documents to generate a summary highlighting advancements in the use of Video Transformers for action recognition.
    #E3 = ChatBot[Summarize #E1, #E2]

    Example 2:
    Task: Analyze the importance of GAN metrizability for improved performance in generative models.
    Plan: Search Arxiv for recent studies on GAN metrizability and its impact on generative model performance using the Searcher agent.
    #E1 = Searcher[GAN metrizability and generative models]
    Plan: Retrieve any embedded research on GAN metrizability from the vector database using the RagSearcher agent.
    #E2 = RagSearcher[GAN metrizability in generative models]
    Plan: Summarize findings on why metrizability is significant for GAN performance based on the retrieved papers.
    #E3 = ChatBot[Summarize #E1, #E2]

    Task: {task}
    """

    rag_guidance = """
    You will receive guidance from a knowledge base checking agent to determine if the RagSearcher step is required.
    'Relevant': Include RagSearcher step in the plan.
    'Non-relevant': Do not include RagSearcher step in the plan.

    Gudiance from knowledge checking agent: Task is '{knowledge_chain_answer}' to the knowledge in the database.

    Please strictly output the plan only. Do not try to solve the query or task yourself Your only task is to plan.
    """
# You will be given a simple question or sub-questions separated by commas that are decomposed from the original question.
#     A subquestion typically corresponds to one individual step or action in the overall plan. 
#    It is a focused inquiry that breaks down a larger task or process into smaller, more manageable parts. 
#    When addressing the question or subquestions, follow the detailed instructions below.

#    - If past messages contain partial results, instructions, or clarifications, ensure they are factored into the plan and appropriately referenced in queries.
#    - Use the RagSearcher for retrieval from pre-embedded research papers in vectorDB, the Searcher for broader academic or web searches, and other agents as needed for task-specific actions.


    trimmer = trim_messages(
    max_tokens=5984,
    strategy="last",
    token_counter=tiktoken_counter,
    include_system=True,
    allow_partial=False,)

    prompt_template = ChatPromptTemplate(
        [
            ('system', few_shot_rewoo),
            ('system', rag_guidance),
            (MessagesPlaceholder('messages'))
        ]
    )

    planner = prompt_template | trimmer | llm | (lambda x: x.content)
    return planner


def assign_chat_topic(llm):

    template = """
        "You are an expert in assigning concise topics to conversations. The user provides their focus area, "
        "and you assign a relevant topic in 5 words or less. "
        "Here is the user's input:\n\n"
        f"{user_input}\n\n"
        "What is the best topic for this conversation? Provide only the topic without any extra text."
    """

    prompt_template = ChatPromptTemplate.from_template(template=template)

    assign_chat_topic_chain = prompt_template | llm | (lambda x: x.content)

    return assign_chat_topic_chain


def memory_decision_chain(llm):
    template = """
    You are an assistant tasked with determining whether the user explicitly wants you to memorize something. If so, your goal is to save this information for future reference.

    Instructions:
    - Only classify the action as "memorize" if the user explicitly asks you to remember or store something.
    - Common phrases to look for include: "remember", "memorize", "store", "keep in memory", "note down".
    - Ignore hypothetical, general, or unrelated references to memory, such as "How can I memorize this for an exam?" or "What helps with remembering tasks?"
    - If the user asks you to remember something, rephrase their request in a clear and concise format. For example, if the user says, "Can you remember that I like vanilla ice cream?", the assistant should store: "user likes vanilla ice cream."
    - If the input does not meet these criteria, classify the action as "none" and leave the content empty.

    Examples:

    Input: "Remember that my dog's name is Max."
    Output: 
        "action": "memorize", 
        "content": "user's dog's name is Max."

    Input: "How can I memorize vocabulary faster?"
    Output: 
        "action": "none", 
        "content": ""

    Input: "Can you remember that I like vanilla ice cream?"
    Output: 
        "action": "memorize", 
        "content": "user likes vanilla ice cream."

    Input: "What helps with remembering daily tasks?"
    Output: 
        "action": "none", 
        "content": ""

    Input: {user_input}
    """


    # Structured output based on the LLM response
    structured_output_llm = llm.with_structured_output(DecideMemorise)
    
    # Using the memory prompt template to guide the model's reasoning
    memory_prompt = ChatPromptTemplate.from_template(template)
    
    # Chain to run the memory decision task
    memory_chain = (
        memory_prompt
        | structured_output_llm
    )
    
    return memory_chain


class knowledgeBaseCheck(TypedDict):
  evaluation: Literal['Relevant', 'Non-relevant']

def check_knowledge_base_chain(llm):
  template='''
    You will be given a set of research papers present in the vector database. Each paper will include the title, authors, and a brief description of the paper based on its abstract.

    Additionally, you will also have access to recent conversation history. Using this information, your task is to evaluate whether the given user query relates to the knowledge present in the vector database. This will help the planner agent decide whether to include the RAG search step in the plan.

    Papers present in the database as vector embeddings: {data}

    User query: {query}

    Recent messages: {recent_messages}
  '''

  structured_output_llm = llm.with_structured_output(knowledgeBaseCheck)

  knowledge_check_prompt = ChatPromptTemplate.from_template(template)
  
  knowledge_check_chain = knowledge_check_prompt | structured_output_llm | (lambda x : x['evaluation'])

  return knowledge_check_chain


class PaperMetadata(TypedDict):
    title: Annotated[str, "The title or heading of the research paper."]
    authors: Annotated[List[str], "The authors of the research paper."]
    publish_date: Annotated[Optional[str], "The publication date of the research paper in the format YYYY-MM-DD."]
    description: Annotated[str, "A concise description (2-3 sentences) summarizing the content of the research paper."]

def basic_metadata_extraction_chain(llm):
    """
    Creates a metadata extraction chain using a given language model.

    Args:
        llm: The language model to be used for metadata extraction.

    Returns:
        Runnable: A chain that extracts the title, authors, publish date, and description from the text.
    """
    # Template for the LLM
    template = '''
    You will be provided with the initial content of a research paper. Your task is to extract the following metadata accurately. 
    If you are uncertain about the answer or cannot find the information, populate the field with a suitable comment explaining why the information is unavailable.

    The details to extract are as follows:
    1. **Title of the paper:** Provide the exact title as it appears in the content. If the title is unclear, state "Title not found in the provided content."
    2. **List of authors:** Extract all authors listed. If no authors are mentioned, state "Authors not mentioned in the provided content."
    3. **Publication date:** Provide the date in the format YYYY-MM-DD. If no date is found, state "Publication date not available."
    4. **Description:** Summarize the paper’s content in strictly maximum of 1-2 concise sentences. If the description cannot be inferred, state "Insufficient information to provide a description."

    Use this structure to ensure clarity and completeness. If you need to make assumptions, mention them explicitly in the output.

    Content:
    {content}
    '''

    # Configure the LLM with structured output
    structured_output_llm = llm.with_structured_output(PaperMetadata)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ('system', template),
            # MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Define additional processing steps if needed (e.g., trimming input text)
    chain = prompt_template | trimmer_first | structured_output_llm

    return chain
