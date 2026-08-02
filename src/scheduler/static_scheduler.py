class StaticScheduler:

    def schedule(self, graph):

        return {
            "Tokenizer": "CPU",
            "LLM": "CPU",
            "Decoder": "CPU"
        }