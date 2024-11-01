from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from prompts import prompts
from prompt_matcher import PromptMatcher

app = FastAPI()

matcher = PromptMatcher()
matcher.store_prompts(prompts)


class PromptRequest(BaseModel):
    prompt: str
    prompt_id: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def read_health():
    return {"status": "healthy"}


@app.post("/match_prompt")
def match_prompt(request: PromptRequest):
    try:
        result = matcher.match_prompt(request.prompt, request.prompt_id)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
