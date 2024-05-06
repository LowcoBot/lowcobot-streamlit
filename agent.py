import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai.chat_models import ChatOpenAI

from tools.allTools import allTools
from tools.prompt import prompt


class LLMAgent:
    def __init__(self) -> None:
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        self.executor = self.__build_executor()

    def __build_executor(self) -> AgentExecutor:
        load_dotenv()

        # If you want to use Mistral instead
        if False:
            mistral_api_key = os.getenv("MISTRAL_API_KEY")
            if mistral_api_key is None:
                raise Exception("MISTRAL_API_KEY not set in .env file")
            model = ChatMistralAI(
                model="open-mixtral-8x7b", mistral_api_key=mistral_api_key
            )
            pass

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise Exception("OPENAI_API_KEY not set in .env file")
        is_gpt_35 = False
        modelName = "gpt-3.5-turbo-16k-0613" if is_gpt_35 else "gpt-4-0125-preview"
        model = ChatOpenAI(model=modelName, api_key=openai_api_key)
        tools = allTools
        agent = create_structured_chat_agent(model, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
        )

    def run_question(self, question: str) -> None:
        print(self.memory.buffer_as_messages)
        response = self.executor.invoke(
            {
                "input": question,
                "chat_history": self.memory.buffer_as_messages,
            }
        )
        response_output = response["output"]
        return response_output
