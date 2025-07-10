from dotenv import load_dotenv
from pprint import pprint
import re
import os
from datetime import datetime
from functools import partial

# Langchain & Langgraph imports
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, trim_messages
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
# API tools
from .team_tools import tavily_search_tool, arxiv_search_tool, repl_tool
from .qdrant_cloud_ops import initialize_selfquery_retriever, qdrant_vector_store
from .store_memory_ops import extract_text_until_introduction, extract_paper_metadata_with_chain, prepare_system_message

# LLM Chains and other modules
from .llm_chains import (
    decomposition_chain,
    requires_decomposition,
    rephrase_chain,
    get_plan_chain,
    assign_chat_topic,
    memory_decision_chain,
    check_knowledge_base_chain,
    basic_metadata_extraction_chain
)

# Token Counter
from .token_counter import tiktoken_counter

# Typing
from typing_extensions import TypedDict, List, Optional, Union, Dict, Literal, Annotated
import operator
from .pre_chat import examples
load_dotenv()

# chains.py

# llm = ChatGroq(model='llama-3.3-70b-versatile')
llm = ChatDeepSeek(model='deepseek-chat', temperature=0.2)

assign_chat_topic_chain = assign_chat_topic(llm=llm)



trimmer = trim_messages(
    max_tokens=5984,
    strategy="last",
    token_counter=tiktoken_counter,
    include_system=True,
    allow_partial=False,
)

trimmer_first = trim_messages(
    max_tokens=5984,
    strategy="first",
    token_counter=tiktoken_counter,
    # include_system=False,
    allow_partial=True,
)

qdrant_retriever = initialize_selfquery_retriever(llm, qdrant_vector_store=qdrant_vector_store, examples=examples)
qdrant_retriever_tool = qdrant_retriever.as_tool(
    name="retrieve_research_paper_texts",
    description="Search and return information from the vector database containing texts of several research papers, and scholarly articles. optionally, align the search process based on pdf name (.pdf file) if given.",
)

# react_agent = create_react_agent(model=llm, checkpointer=MemorySaver(),tools=[qdrant_retriever_tool, arxiv_search_tool, tavily_search_tool], state_modifier="You are a helpful research assistant. Help user to the best of your abilities. Provide concise but accurate and up to point answers. As of now you have these tools in your arsenal: qdrant_retriever_tool (content retrieval from vector database), arxiv_search_tool (search research papers), tavily_search tool (internet search). If you do not know the answer, then simply say 'I don't know. If you need clarification on what exactly user wants, then ask the user again. If you know the answer to user's query then answer yourself, else you can also rely on tools you have.")



tools_for_agent = [qdrant_retriever_tool, arxiv_search_tool, tavily_search_tool]

metadata_extraction_chain = basic_metadata_extraction_chain(llm=llm)


# # decomposer_chain = decomposition_chain(llm=llm)
# # check_query_chain = requires_decomposition(llm=llm)
# # rephraser_chain = rephrase_chain(llm=llm)
# planner_chain = get_plan_chain(llm=llm)
# # assign_topic_chain = assign_chat_topic(llm=llm)
# check_knowledge_base = check_knowledge_base_chain(llm=llm)


# class CoreState(TypedDict):
#   messages: Annotated[List[BaseMessage], operator.add]
#   passes: Dict[str, str]
#   call_next: str
#   current_task: str

# class ResearchTeamState(TypedDict):
#   messages: Annotated[List[BaseMessage], operator.add]    
#   task: str
#   plan_string: str
#   steps: List
#   results: dict
#   result: str

#   current_task: str

# Workers = Literal['research_team', 'LLM']

# members = ", ".join(['research_team', 'LLM'])

# supervisor_prompt = f'You are a supervisor and your task is to delegate to one of the members from: {members}.' + '''

#     ### Guidelines:
#     1. **Casual or General Queries**: 
#     - If the query is casual, general, or involves providing information that does not require specific investigation, delegate the task to the 'LLM' to answer directly.
#     - Example: If the question is "What is the capital of France?", delegate it to 'LLM' to provide the answer.

#     2. **Task-Specific Queries**:
#     - **Research Tasks**: If the query involves research, new findings, or needs further investigation (e.g., "Can you research the latest advancements in AI?" or "Tell me more about quantum computing?"), delegate the task to the **research_team**. **Do not** answer these types of questions directly with the 'LLM'.
#     - Example: For a question like "Can you research the latest advancements in AI?", delegate it to 'research_team' to gather research findings.

#     3. **Output Format**:
#     - Your output should be formatted as a dictionary with workers as keys and their respective tasks as values.
#     - Each key represents the worker (either 'LLM' or 'research_team') who will be responsible for completing the task.
#     - Each value represents the task assigned to that worker.

#     Example output format:


#     {{
#         'LLM': 'Answer the question: "What is the capital of France?"',
#         'research_team': 'Research the latest advancements in AI.'
#     }}


#     ### Warnings:
#     - Make sure that tasks are assigned logically:
#     - LLM should handle simple informational queries.
#     - Research_team should handle queries requiring detailed research or new findings.
#     - Ensure the tasks are clearly assigned and that the workers can perform them based on their capabilities.
#     - If the query is neither casual nor requiring research, it may not need to be delegated. Handle such edge cases appropriately.
#     - If you're unsure about the user's request or lack context, always route to 'LLM' to ask for clarification instead of making assumptions or taking random actions.
#     ---

#     ### Examples:

#     1. **Casual Query Example**:
#     **Question**: "What is the tallest mountain in the world?"
#     **Output**:

#     {{
#         'LLM': 'Answer the question: "What is the tallest mountain in the world?"'
#     }}


#     2. **Research Query Example**:
#     **Question**: "Can you research the latest advancements in quantum computing?"
#     **Output**:

#     {{
#         'research_team': 'Research the latest advancements in quantum computing.'
#     }}


#     3. **Mixed Query Example**:
#     **Question**: "Tell me about the capital of Japan, and also research the impacts of artificial intelligence on education."
#     **Output**:

#     {{
#         'LLM': 'Answer the question: "What is the capital of Japan?"',
#         'research_team': 'Research the impacts of artificial intelligence on education.'
#     }}


#     Question: {question}
# '''





# def make_supervisor_node(llm: BaseChatModel, members: List[str], prompt_for_supervisor: str):
#   options = members
#   system_prompt = prompt_for_supervisor

#   class SequenceCreator(TypedDict):
#       """Defines the sequence of workers and their tasks."""
#       passes: Dict[Workers, str]
#       def __init__(self, workers_tasks: Dict[Workers, str]):
#           self.passes = workers_tasks

#   def supervisor_node(state):
#       template = ChatPromptTemplate([
#       ('system', system_prompt),
#       MessagesPlaceholder(variable_name="messages"),
#       ])
#       print('\nQUESTION: ',state['messages'][-1].content, '\n')
#       structured_llm_output = llm.with_structured_output(SequenceCreator)

#       response_chain = template | trimmer | structured_llm_output

#       response = response_chain.invoke({'question': state['messages'][-1].content, 'messages': state['messages']})
#       return {'passes': response['passes']}


#   return supervisor_node

# core_supervisor_node = make_supervisor_node(llm=llm, members=['research_team', 'LLM'], prompt_for_supervisor=supervisor_prompt)

# def router(state: CoreState) -> Command[Literal["LLM", 'research_team', END]]:
#     if not state['passes']:
#         return Command(goto="__end__", update={'call_next': '', 'passes': {}, 'current_task': ''})

#     # print('\nWORKING LIST: ', state['passes'], '\n')

#     worker, task_value = next(iter(state['passes'].items()))
#     print(f'\nNEXT WORKER: {worker}, TASK: {task_value}')

#     del state['passes'][worker]

#     return Command(goto=worker, update={'call_next': worker, 'current_task': task_value, 'passes': state['passes']})

# async def get_plan(state:ResearchTeamState) -> Command[Literal["agent_exec", END]]:
#     try:
#         regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"

#         task = state['current_task']
#         messages = state['messages']
#         print('\nCURRENT PLAN QUERY:', task)

#         metadata_info = await prepare_system_message

#         rag_guidance = check_knowledge_base.ainvoke({'query': task, 'data': metadata_info, 'recent_messages': messages[-12:]})
#         plan = planner_chain.invoke({'task': task, 'knowledge_chain_answer': rag_guidance, 'messages': messages})

#         # print('\nMETADATA INFO: ',metadata_info, '\n')
#         print('\nRAG GUIDANCE: ',rag_guidance, '\n')

#         # print(plan, '\n', f'{"-" * 56}', '\n')

#         matches = re.findall(regex_pattern, plan)
#         print(f'\n PLAN: {plan}\n')
#         return Command(update={"steps": matches, "plan_string": plan, 'task': task}, goto='agent_exec')

#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return Command(update={"steps": [], "plan_string": "", 'task': ''}, goto=END)


  
# rag_agent = create_react_agent(llm, tools=[qdrant_retriever_tool])
# search_agent = create_react_agent(llm, tools=[tavily_search_tool])
# arxiv_agent = create_react_agent(llm, tools=[arxiv_search_tool])
# code_agent = create_react_agent(llm, tools=[repl_tool])


# def code_node(state: ResearchTeamState):
#     return {"messages": [HumanMessage(content=state['messages'][-1].content, name="Coder")]}

# def retriever_node(state: ResearchTeamState):
#     return {"messages": [HumanMessage(content=state['messages'][-1].content, name="Retriever")]}

# def search_node(state: ResearchTeamState):
#     return {"messages": [HumanMessage(content=state['messages'][-1].content, name="Searcher")]}

# def arxiv_search_node(state: ResearchTeamState):
#     return {"messages": [HumanMessage(content=state['messages'][-1].content, name="ArXivSearcher")]}


# def _get_current_task(state: ResearchTeamState):
#     if "results" not in state or state["results"] is None:
#         return 1
#     if len(state["results"]) == len(state["steps"]):
#         return None
#     else:
#         return len(state["results"]) + 1



# def agent_exec(state: ResearchTeamState):
#     """Worker node that executes the agents accordingly for a given plan."""
#     try:

#         _results = (state["results"] or {}) if "results" in state else {}
#         _step = _get_current_task(state)
#         step_desc, step_name, agent, agent_input = state["steps"][_step - 1]

#         # Replace placeholders in agent_input with corresponding results
#         for k, v in _results.items():
#             agent_input = agent_input.replace(k, v) 

#         # Dynamically select the agent function based on the agent name
#         if agent == "RagSearcher":
#             result = rag_agent.invoke(retriever_node(state))['messages'][-1].content
#         elif agent == "Searcher":
#             result = search_agent.invoke(search_node(state))['messages'][-1].content
#         elif agent == "ChatBot":
#             result = llm.invoke(agent_input)  # Assuming LLM invocation does not need message formatting
#         elif agent == "Coder":
#             result = code_agent.invoke(code_node(state))['messages'][-1].content
#         elif agent == "ArXivSearcher":
#             result = arxiv_agent.invoke(arxiv_search_node(state))['messages'][-1].content
#         else:
#             raise ValueError(f"Unknown agent type: {agent}")

#         if result is None:
#             raise ValueError(f"Agent {agent} did not return a result for step {step_name}")

#         # Store the result in the _results dictionary
#         _results[step_name] = str(result)

#         return {"results": _results}
    
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return {'plan_string': "", 'steps': [], 'results': {}, 'result': ""}

# solve_prompt = """
# We have created a detailed step-by-step Plan to solve the given task and obtained corresponding answers from agents for each step in the Plan. 
# Use these agent-provided answers as Evidence to craft a clear, comprehensive, and cohesive response.

# Here are some recent few messages to give you a context of the conversation:
# {recent_messages}

# Plan:  
# {plan}

# Using the Evidence from the answers provided for each step in the Plan, solve the given task:  
# Task: {task}

# """



# def solve(state: ResearchTeamState) -> Command[Literal[END]]:
#     try:
#         plan = ""
#         for _plan, step_name, agent, agent_input in state["steps"]:
#             _results = (state["results"] or {}) if "results" in state else {}
#             for k, v in _results.items():
#                 agent_input = agent_input.replace(k, v)
#                 step_name = step_name.replace(k, v)
#             plan += f"Plan: {_plan}\n{step_name} = {agent}[{agent_input}]"

#         print('\nplan:', plan)
        
#         prompt = solve_prompt.format(plan=plan, task=state['current_task'], recent_messages=state['messages'][-12:])
#         result = llm.invoke(prompt)
        
#         return Command(
#             update={
#                 "result": result.content,
#                 'messages': [result],
#                 'results': {}, 
#                 'steps': [],
#                 'task': '',
#                 'plan_string': '',
#                 'current_task': ''
#             },
#             goto=END,
#         )
#         # return {
#         #         "result": '',
#         #         'messages': [result],
#         #         'results': {}, 
#         #         'steps': [],
#         #         'task': '',
#         #         'plan_string': '',
#         #         'current_task': '',
#         #     },
    
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return Command(update={'task': '', 'plan_string': '', 'steps': [], 'results': {}, 'result': "", 'current_task': ''}, goto=END)
#         # return e


# def _route(state: ResearchTeamState):
#     _step = _get_current_task(state)
#     if _step is None:
#         return "solve"
#     else:
#         return "agent_exec"

# research_graph = StateGraph(ResearchTeamState)

# # graph.add_node('decompose_or_rephrase', decompose_or_rephrase)
# research_graph.add_node('get_plan', get_plan)
# research_graph.add_node("agent_exec", agent_exec)
# research_graph.add_node("solve", solve)




# # # research_graph.add_edge(START, 'decompose_or_rephrase')
# research_graph.add_edge(START, 'get_plan')
# # # research_graph.add_edge('decompose_or_rephrase', 'get_plan')
# # research_graph.add_edge('get_plan', 'agent_exec')
# research_graph.add_conditional_edges("agent_exec", _route)

# research_graph = research_graph.compile()


# def call_llm(state: CoreState):
#   task = state['current_task']

#   llm_prompt = """
# You are a helpful assistant with access to the current conversation history, but only up to the context window size. Your task is to assist to the best of your abilities, based on the information provided. Please consider the context in which the task is given and strive to perform it as effectively and satisfactorily as possible. 

# Remember:
# - Use the conversation history to understand the context of the task.
# - Answer concisely but thoroughly based on the information available.

# Task: {task}
# """



#   template = ChatPromptTemplate(
#     [
#       ('system', llm_prompt),
#       MessagesPlaceholder('messages')
#     ]
#   )
#   llm_chain = template | trimmer | llm
#   llm_response = llm_chain.invoke({'task': task, 'messages':state['messages']})
#   return {'messages': [llm_response]}

# core_graph = StateGraph(CoreState)

# core_graph.add_node('core_supervisor', core_supervisor_node)
# core_graph.add_node('research_team', research_graph)
# core_graph.add_node('LLM', call_llm)
# core_graph.add_node('router', router)

# core_graph.add_edge(START, 'core_supervisor')
# core_graph.add_edge('core_supervisor', 'router')
# core_graph.add_edge('LLM', 'router')
# core_graph.add_edge('research_team', 'router')




# core_graph = core_graph.compile(checkpointer=memory)
