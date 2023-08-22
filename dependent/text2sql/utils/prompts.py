from pydantic import BaseModel
from typing import List, Tuple, Iterable, Union, Optional

class Prompt(BaseModel):
    #template: str = "here is a schema: {schema}. write a sq query to answer the following question: {question}. query:"
    template: str = "given a database schema( {schema}) | wirte a sql script to answer({question}) | query: "
            
    def __call__(
            self,
            schema: Union[Iterable[str], str] = ...,
            question: Union[Iterable[str], str] = ...,
            query: Optional[Union[Iterable[str], str]] = None
    ):
        if type(schema) is str or type(question) is str:
            assert type(question) is str and type(question) is str
            if query is not None:
                assert type(query) is str
            return self.template.format(schema = schema, question = question) + (query if query is not None else "")
        else:
            assert len(question) == len(schema)
            inputs = []
            for i in range(len(question)):
                inputs.append(self.template.format(schema = schema, question = question))
            if query is not None:
                assert len(question) == len(schema) == len(query)
                for i in range(len(query)):
                    inputs[i] += query[i]
            return inputs
     