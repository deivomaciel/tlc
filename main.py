from llama_cpp import Llama
import multiprocessing
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

class LLM:
  def __init__(self, repo_id: str, model_filename: str) -> None:
     self.__repo_id = repo_id
     self.__model_filename = model_filename
     self.__llm = None
     self.model_path = None
     self.__system_threads = multiprocessing.cpu_count()

  def init(self):
    self.__llm = Llama.from_pretrained(
        repo_id=self.__repo_id,
        filename=self.__model_filename,
        n_batch=128,
        n_ctx=1024,
        n_threads=self.__system_threads,
        cache=True,
        use_mmap=True,
        use_mlock=False,
        n_gpu_layers=0,
        verbose=False
    )

    self.model_path = self.__llm.model_path

  def getResponse(self, prompt: str, max_tokens=80):
    condition = ' com cada linha separada por uma quebra de linha (\\n)'
    system_prompt = """Você sempre traz informações verdadeiras e científicas. com cada linha separada por uma quebra de linha (\\n)
      Se não souber algo, diga que não tem certeza.
    """

    # response = self.__llm.create_completion(
    #     prompt= f"{system_prompt}\nUsuário: {prompt}\nResposta:",
    #     max_tokens=max_tokens,
    #     temperature=0.3,
    #     top_p=0.9,
    #     repeat_penalty=1.15,
    #     echo=False,
    #     stream=True
    # )

    # return response["choices"][0]["text"]

    for chunks in self.__llm.create_completion(
        prompt= f"{system_prompt}\nUsuário: {prompt}\nResposta:",
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.15,
        echo=False,
        stream=True
    ): print(chunks["choices"][0]["text"], end="", flush=True)

llm = LLM("unsloth/gemma-3-270m-it-GGUF", "gemma-3-270m-it-Q5_K_M.gguf")
llm.init()

app = Flask(__name__)
CORS(app)

@app.route('/llm', methods=['POST'])
def getLLMResponse():
  data = request.get_json()
  prompt = data['prompt']

  response = llm.getResponse(prompt)

  return jsonify({'message': response})

if __name__ == '__main__':
    app.run(port=3001)
