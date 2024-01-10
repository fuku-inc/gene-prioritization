import yaml
import pandas as pd
from typing import List
import openai
import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
import argparse


def main(config, curated, rnaseq, log):
    with open(f'/app/{config}', 'r') as yml:
        config = yaml.safe_load(yml)
    apikey = config["OPENAI_API_KEY"]
    model = config["model"]

    chat = ChatOpenAI(model = model, openai_api_key = apikey, temperature=0.0)

    # read a repaired tsv file
    df = pd.read_csv(f'/app/{curated}', sep='\t')
    # extract summaries
    list_s = df.iloc[:, 3::3].values.ravel().tolist()
    str_s = "".join(list_s)
    list_function = df['gene_function'].tolist()
    str_function = "".join(list_function)

    # set prompts
    system_settings = f"""
    {str_function}
    {str_s}
    Based on the summary provided above, please answer the question.
    """

    rnaseq_prompt = config["rnaseq_prompt"]
    with open(f'/app/{rnaseq}', "r") as f:
        textualized_rnaseq = f.readlines()
    final_prompt = f"""
    {rnaseq_prompt}
    {textualized_rnaseq}
    """

    output_list = []

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_settings),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    print(prompt)
    with open(f'./logs/{log}', 'a') as f:
        print(prompt, file=f)
    conversation = ConversationChain(
        memory=ConversationBufferMemory(return_messages=True),
        prompt=prompt,
        llm=chat)
    with get_openai_callback() as cb:
        output1 = conversation.predict(input=config["selection_prompt"])
        print(cb)
        with open(f'./logs/{log}', 'a') as f:
            print(cb, file=f)
            print(output1, file=f)
        output_list.append(output1)

        output2 = conversation.predict(input=final_prompt)
        print(cb)
        with open(f'./logs/{log}', 'a') as f:
            print(cb, file=f)
            print(output2, file=f)
        output_list.append(output2)

        output3 = conversation.predict(input="Could you summarize the key conclusions you have drawn from the conversation so far?")
        print(cb)
        with open(f'./logs/{log}', 'a') as f:
            print(cb, file=f)
            print(output3, file=f)
        output_list.append(output3)

    print(f"score_selection_list:{output_list}")
    with open(f'./logs/{log}', 'a') as f:
        print(output_list, file=f)

    with open(config["output_path_for_selection"], 'w') as f:
        for item in output_list:
            f.write(item+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--curated")
    parser.add_argument("--rnaseq")
    parser.add_argument("--log")
    args = parser.parse_args()
    main(args.config, args.curated, args.rnaseq, args.log)
