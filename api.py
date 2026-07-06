from fastapi import FastAPI
from piton import answer_question

answer_question("Is JavaScript a object oriented programing language?")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

