{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "80826955",
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install langgraph"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "176a632c",
   "metadata": {},
   "source": [
    "StateGraph: https://langchain-ai.github.io/langgraphjs/reference/classes/langgraph.StateGraph.html\n",
    "A graph whose nodes communicate by reading and writing to a shared state. Each node takes a defined State as input and returns a Partial<State>.\n",
    "    \n",
    "TypedDict: https://docs.python.org/3/library/typing.html#typing.TypedDict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "16799561",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: langchain_core in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (0.3.51)\n",
      "Requirement already satisfied: jsonpatch<2.0,>=1.33 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from langchain_core) (1.33)\n",
      "Requirement already satisfied: pydantic<3.0.0,>=2.5.2 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from langchain_core) (2.10.6)\n",
      "Requirement already satisfied: packaging<25,>=23.2 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from langchain_core) (24.2)\n",
      "Requirement already satisfied: typing-extensions>=4.7 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from langchain_core) (4.12.2)\n",
      "Requirement already satisfied: langsmith<0.4,>=0.1.125 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from langchain_core) (0.3.24)\n",
      "Requirement already satisfied: PyYAML>=5.3 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from langchain_core) (6.0)\n",
      "Requirement already satisfied: tenacity!=8.4.0,<10.0.0,>=8.1.0 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from langchain_core) (9.0.0)\n",
      "Requirement already satisfied: jsonpointer>=1.9 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from jsonpatch<2.0,>=1.33->langchain_core) (3.0.0)\n",
      "Requirement already satisfied: zstandard<0.24.0,>=0.23.0 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from langsmith<0.4,>=0.1.125->langchain_core) (0.23.0)\n",
      "Requirement already satisfied: requests-toolbelt<2.0.0,>=1.0.0 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from langsmith<0.4,>=0.1.125->langchain_core) (1.0.0)\n",
      "Requirement already satisfied: httpx<1,>=0.23.0 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from langsmith<0.4,>=0.1.125->langchain_core) (0.28.1)\n",
      "Requirement already satisfied: requests<3,>=2 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from langsmith<0.4,>=0.1.125->langchain_core) (2.32.3)\n",
      "Requirement already satisfied: orjson<4.0.0,>=3.9.14 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from langsmith<0.4,>=0.1.125->langchain_core) (3.10.15)\n",
      "Requirement already satisfied: idna in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from httpx<1,>=0.23.0->langsmith<0.4,>=0.1.125->langchain_core) (3.2)\n",
      "Requirement already satisfied: httpcore==1.* in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from httpx<1,>=0.23.0->langsmith<0.4,>=0.1.125->langchain_core) (1.0.7)\n",
      "Requirement already satisfied: anyio in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from httpx<1,>=0.23.0->langsmith<0.4,>=0.1.125->langchain_core) (4.8.0)\n",
      "Requirement already satisfied: certifi in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from httpx<1,>=0.23.0->langsmith<0.4,>=0.1.125->langchain_core) (2021.10.8)\n",
      "Requirement already satisfied: h11<0.15,>=0.13 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->langsmith<0.4,>=0.1.125->langchain_core) (0.14.0)\n",
      "Requirement already satisfied: annotated-types>=0.6.0 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from pydantic<3.0.0,>=2.5.2->langchain_core) (0.7.0)\n",
      "Requirement already satisfied: pydantic-core==2.27.2 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from pydantic<3.0.0,>=2.5.2->langchain_core) (2.27.2)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from requests<3,>=2->langsmith<0.4,>=0.1.125->langchain_core) (2.0.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from requests<3,>=2->langsmith<0.4,>=0.1.125->langchain_core) (2.3.0)\n",
      "Requirement already satisfied: sniffio>=1.1 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from anyio->httpx<1,>=0.23.0->langsmith<0.4,>=0.1.125->langchain_core) (1.2.0)\n",
      "Requirement already satisfied: exceptiongroup>=1.0.2 in /Users/swatiphartiyal/opt/anaconda3/lib/python3.9/site-packages (from anyio->httpx<1,>=0.23.0->langsmith<0.4,>=0.1.125->langchain_core) (1.2.2)\n"
     ]
    }
   ],
   "source": [
    "!pip install langchain_core"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a8693430",
   "metadata": {},
   "outputs": [],
   "source": [
    "CH_HOST = 'http://localhost:8123' # default address \n",
    "import requests\n",
    "\n",
    "def get_clickhouse_data(query, host = CH_HOST, connection_timeout = 1500):\n",
    "  r = requests.post(host, params = {'query': query}, \n",
    "    timeout = connection_timeout)\n",
    "  if r.status_code == 200:\n",
    "      return r.text\n",
    "  else: \n",
    "      return 'Database returned the following error:\\n' + r.text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d862d69d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from typing import TypedDict, Annotated\n",
    "from langchain_core.messages import AnyMessage\n",
    "import operator\n",
    "\n",
    "\n",
    "class State(TypedDict):\n",
    "    messages: Annotated[list[AnyMessage], operator.add]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4fc6119c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_core.tools import tool\n",
    "\n",
    "@tool\n",
    "def execute_sql(query: str) -> str:\n",
    "    \"\"\"Returns the result of SQL query execution\"\"\"\n",
    "    return get_clickhouse_data(query)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dc34df5f",
   "metadata": {},
   "source": [
    "invoke: https://python.langchain.com/docs/concepts/tool_calling/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "867d8ff7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langgraph.graph import StateGraph,END\n",
    "from langchain_core.messages import SystemMessage,ToolMessage\n",
    "\n",
    "class SQLAgent:\n",
    "    def __init__(self, model, tools, system_prompt=\"\"):\n",
    "        self.system_prompt = system_prompt\n",
    "        self.tools = {t.name: t for t in tools}\n",
    "        self.model = model.bind_tools(tools)\n",
    "\n",
    "        \n",
    "        graph = StateGraph(State)\n",
    "    \n",
    "        graph.add_node(\"llm\", self.call_llm)\n",
    "        graph.add_node(\"function\", self.execute_function)\n",
    "    \n",
    "    \n",
    "        #add dictionary with N keys (where N is the number of possible outputs \n",
    "        #from the conditional check function, in our case two) and values \n",
    "        #as the associated nodes that will be redirected to\n",
    "    \n",
    "        graph.add_conditional_edges(\n",
    "            \"llm\",\n",
    "            self.is_a_tool_required,\n",
    "            {True: \"function\", \n",
    "             False: END}\n",
    "            )\n",
    "    \n",
    "        graph.add_edge(\"function\", \"llm\")\n",
    "    \n",
    "        graph.set_entry_point(\"llm\")\n",
    "    \n",
    "        self.graph = graph.compile()\n",
    "        \n",
    "    \n",
    "    def is_a_tool_required(self, state: State):\n",
    "        tool_calls = state['messages'][-1].tool_calls\n",
    "        return len(tool_calls) > 0\n",
    "        \n",
    "    \n",
    "    def call_llm(self, state: State):\n",
    "        messages = state['messages']\n",
    "        if self.system_prompt:\n",
    "            messages = [SystemMessage(content=self.system_prompt)] + messages\n",
    "        message = self.model.invoke(messages)\n",
    "        return {'messages': [message]}\n",
    "    \n",
    "    def execute_function(self, state:State):\n",
    "        tool_calls = state['messages'][-1].tool_calls\n",
    "        results = []\n",
    "        for t in tool_calls:\n",
    "            print(f\"Calling: {t}\")\n",
    "            if not t['name'] in self.tools:      # check for bad tool name from LLM\n",
    "                print(\"\\n ....bad tool name....\")\n",
    "                result = \"bad tool name, retry\"  # instruct LLM to retry if bad\n",
    "            else:\n",
    "                result = self.tools[t['name']].invoke(t['args'])\n",
    "            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))\n",
    "        print(\"Back to the model!\")\n",
    "        return {'messages': results}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4ddcd108",
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = '''You are a senior expert in SQL and data analysis. \n",
    "So, you can help the team to gather needed data to power their decisions. \n",
    "You are very accurate and take into account all the nuances in data.\n",
    "Your goal is to provide the detailed documentation for the table in database \n",
    "that will help users. Please, keep your answers concise and straight to the point.'''\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db9ceeb4",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "model = ChatOpenAI(model=\"gpt-4o-mini\")\n",
    "doc_agent = SQLAgent(model, tools=[execute_sql], system_prompt=prompt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "120e7878",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_core.messages import HumanMessage\n",
    "\n",
    "messages = [HumanMessage(content=\"What info do we have in ecommerce_db.users table?\")]\n",
    "result = doc_agent.graph.invoke({\"messages\": messages})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20ec7b60",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(result['messages'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5b2784e4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
