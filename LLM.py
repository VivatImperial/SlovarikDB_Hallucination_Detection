from llama_cpp import Llama


class LLM_predictor:
    def __init__(self) -> None:
        self.llm = Llama(
            model_path="llama_w\model-q4_K.gguf",
            n_parts=1,
            n_gpu_layers=-1,
            n_ctx=8192
        )
        
        self.system_prompt_start = {
            "role": "system",
            "content": """Ты — Сайга, русскоязычный автоматический ассистент. 
                       Ты разговариваешь с людьми и помогаешь им. 
                       Твоя задача - находить ошибки в ответах на вопросы, заданных по конкретному тексту"""
        }

    def get_question(self, text: str, question: str, answer: str, flag: int) -> str:
        task = ""
        if flag == 0:
            task = "Твоя задача: проверять фактологическую правильность указанных численных данных (дат и годов) в ответе на вопрос по тексту. "
        elif flag == 1:
            task = "Твоя задача: проверять фактологическую правильность географических названий (страны, города) в ответе на вопрос по тексту. "

        question_text = f"""
    {task}
    
    Даны текст, вопрос и ответ, 
    вопрос связан с текстом,
    в качестве ответа выведи только одну цифру 
    0 - в случае, если ответ на вопрос был верным и 1 в случае, если ответ на вопрос неверный:
    
    Текст:
    {text}
    
    Вопрос:
    {question}
    
    Ответ:
    {answer}"""

        return question_text

    def predict(self, prompt: list[str], flag: int) -> int:
        message_list = [self.system_prompt_start]
        message_list.append(
            {
                "role": "user",
                "content": self.get_question(
                    prompt[0],
                    prompt[1],
                    prompt[2],
                    flag
                )
            }
        )

        answer = self.llm.create_chat_completion(message_list)

        return int(answer["choices"][0]["message"]["content"].replace("\n", ""))


# if __name__ == "__main__":
#     llm = LLM_predictor()
#     print(
#         llm.predict(
#             [
#                 "'Гималаи или Гималайские горы — горная система в Азии, расположенная к югу от плато Тибет.
#                 Они расположены на территории Пакистана, Индии, Китая и Непала.
#                 Самая высокая гора на Земле гора Эверест (Джомолунгма) — находится в Гималаях.'",
#                 "'Где находится самая высокая гора на Земле?'",
#                 "'В Гималаях.'"
#             ]
#         )
#     )
