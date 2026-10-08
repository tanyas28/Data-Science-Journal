# Chapter 7: Advanced Text Generation Techniques and Tools

This chapter explores the **LangChain** framework.

**Chains** basically help us connect methods and modules.

---

## Chains: Extending the Capabilities of LLMs

The general purpose of chains is to connect an LLM with additional tools to further enhance its performance. Although we can chain the chains as well.

---

### Application 1: Prompt Template

Every time we are building a project around an LLM, or developing an application, we tend to choose different different LLM models. Each model tends to have its specific prompt template most often. Now, every time changing the prompt type is not feasible — the better approach will be to make a **prompt template** that defines the style of the prompt, and then chain it to the LLM.

We create a simple prompt template (for model **Phi-3**):

```python
from langchain import PromptTemplate

template = """<s><|user|>
{input_prompt}<|end|>
<|assistant|>"""
```

Now the chain will look like:

```python
chain1 = prompt | llm
```

To use this defined chain, we use the **`invoke`** function. `"input_prompt"` will be the key used to pass our query.

```python
chain1.invoke(
    {
        "input_prompt": "what is the weather today?",
    }
)
```

---

### Now What If We Want Multiple Prompts?

Let's take an example — suppose we want to generate a story which has three components: its **title**, **characters**, and **summary**.

So for a prompt like "generate a story," we can split it into three prompts: one that creates the title, one that creates the character description, and one that creates the summary. These chains should execute in such a way that only **one** input prompt is required from the user, and the rest of the steps follow **sequentially**.

We can ask the LLM to create a title for a given summary:

```python
from langchain.chains import LLMChain

template = """<s><|user|> create a title for a story about {summary}.<|end|>
<|assistant|>"""

title_p = PromptTemplate(template=template, input_variables=["summary"])

title = LLMChain(llm=llm, prompt=title_p, output_key="title")

title.invoke({"summary": "......"})
```

Similarly, we can make a component that gives the character description using **both** summary and title.

Then we can create the final component that generates the short description/story — it will take summary, title, and character all as input.

The chain will look like:

```python
llm_chain = title | character | story
```

Then we can invoke this chain:

```python
llm_chain.invoke("..........")
```

---

### Application 2: Adding Memory

We can use chains to add **memory** to our LLM. Now that can be of multiple types, such as **conversation buffer** and **conversation summary**.

Now, **conversation buffer memory** can be of two types:
- The simplest one being that you copy the chats into the prompt again — **all** the chats — so the LLM receives the updated prompt each time.
- The other technique is using the **last K chats**, to manage the context window issue.

A conversation-buffer-memory chain will look like:

```python
template = """<s><|user|> current conversation: {chat_history}

{input_prompt}<|end|>
<|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt", "chat_history"])

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")  # for window we add k parameter here {k=2,3..}

llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)
```

Similarly, another type of memory that we can add is **conversation summary**, which keeps a running summary of the conversation happened so far.

We will first define the summary prompt template, then we will define the memory and the chain.

---

### Application 3: Agents — Creating a System of LLMs

So far, using prompt templates in a chain, we were able to make the LLM perform certain tasks — like generating a story, summarizing. We added memory.

But an **agent** is something which, if it has access to tools, can make decisions to use them as and when required and give the output.

But LLMs only know how to process text, hence we need to define to the LLM what each tool does and what are the steps to use it. We can do this by giving instructions in a simple file, and giving tools such as reading the file to the agent.

**ReAct** is a powerful framework for this. It consists of:
1. **Thought**
2. **Action**
3. **Observation**

These steps are followed **iteratively**.

the react template looks something like:

react_temp = """ Answer the fowllowing questions as best as you can. You have acess to following:

{tools}

use the format:

Question: the input 
thought: you should think about what to do.
Action: the is is the action plan where you plan the implementation, and the step should be one of {tool_names}
Action Input: input for the action.
Observation: the result of the action.....(you can repeat thought/action/action input/obeservation N times)

thought: I know final answer.
Final answer: the final answer to the question.

Start!

Question: {input}
Thought: {agent_notepad}

prompt = prompTemplate(
    template = react_temp,
    input_variables=["tools","tool_names","input","agent_notepad"]
)

