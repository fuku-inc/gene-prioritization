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
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
import csv
from io import StringIO
import re
from langchain.callbacks import get_openai_callback
import argparse
import sys



# set the pydantic class
class ResultsForStatements(BaseModel):
    evaluative_comment: str = Field(description="evaluative comment for statement")
    ref: str = Field(description="supporting references for statement")
    score: int = Field(description="score for statement")


class GeneFunctionAndScoreForStatements(BaseModel):
    Gene_Symbol: str = Field(description="gene symbol")
    Gene_Name: str = Field(description="gene name")
    a_brief_summary: str = Field(description="a brief summary of the gene")
    results_for_statements: List[ResultsForStatements]


def main(config, log):
    # read yml file
    with open(f'/app/{config}', 'r') as yml:
        config = yaml.safe_load(yml)  
    apikey = config["OPENAI_API_KEY"]
    model = config["model"]
    score_example = config["score_example"]

    # set the prompt
    system_settings = f"""
    I am going to ask for a given gene to:
    1.Provide the gene's official name
    2.Provide a brief summary of the gene's function.
    3.Give each of the following statements a score from 0 to 10, with 0 indicating no evidence and 10 indicating very strong evidence:
    """

    for i,s in enumerate(config["statements"]):
        system_settings += f"{i+4}.{s}"

    system_settings += f"""
    Scoring criteria:
    0—No evidence found.
    1-3—Very limited evidence.
    4-6—Some evidence, but needs validation or is limited to certain conditions.
    7-8—Good evidence, used or proposed for some applications.
    9-10—Strong evidence, firmly established as a useful biomarker or research target.
    For scores of 4 or above please provide an evaluative comment and up to three key supporting references using as a format: First author, Title, Date, Journal.

    The results must be generated in the following format, using | as a delimiter and on a single line:
    Gene symbol|Gene name|a brief summary|evaluative comment for statement a|supporting references for statement a|score for statement a|and so on for statements b, c, d, e and f.

    just to give an idea of what the output must look like, here is an example for the gene: {score_example}
    """

    # load ChatOpenAI
    chat = ChatOpenAI(model = model, openai_api_key = apikey,temperature=0.0)

    parser = PydanticOutputParser(pydantic_object=GeneFunctionAndScoreForStatements)
    output_fixing_parser = OutputFixingParser.from_llm(
        parser=parser,
        llm=chat
    ) 

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_settings),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    conversation = ConversationChain(
        memory=ConversationBufferMemory(return_messages=True),
        prompt=prompt,
        llm = chat)

    gene_list = config["first_gene_list"]
    print(gene_list)
    with open(f'./logs/{log}', 'a') as f:
        print(gene_list, file=f)

    score_results_list = []
    with get_openai_callback() as cb:
        for gene in gene_list:
            prompt2_3 = f"Now go ahead with the evaluation of this gene: [{gene}]"
            output = conversation.predict(input=prompt2_3)
            print(cb)
            print(output)
            with open(f'./logs/{log}', 'a') as f:
                print(cb, file=f)
                print(output, file=f)
            try:
                result = parser.parse(output)
                print(f"succeed 1st attempt with output_parser")
            except:
                result = output_fixing_parser.parse(output)
                print(f"suceed with auto-fixing_parser")
            print(cb)
            print(result)
            with open(f'./logs/{log}', 'a') as f:
                print(cb, file=f)
                print(result, file=f)

            # parser the dict part 
            data_dict = {}
            for i in result:
                if i[0] != "results_for_statements":
                    data_dict[i[0]] = i[1]
                else:
                    results = i[1:]
                    for x in results:
                        for w,y in enumerate(x):
                            for z in y:
                                data_dict[f"{z[0]}_{w}"] = z[1]  
            print(data_dict)
            with open(f'./logs/{log}', 'a') as f:
                print(data_dict, file=f)
            score_results_list.append(data_dict)

    print(f"score_results_list:{score_results_list}")
    with open(f'./logs/{log}', 'a') as f:
        print(score_results_list, file=f)
    df = pd.DataFrame(score_results_list)
    df.to_csv(config["output_path_for_score"],sep="\t",index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--log")
    args = parser.parse_args()
    main(args.config, args.log)

