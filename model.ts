import { LlamaModel, LlamaContext, LlamaChatSession } from "node-llama-cpp";
import { join } from "path";

function Session (mod) {
  const model = new LlamaModel({
    modelPath: join(__dirname, "models", mod)
  });
  const context = new LlamaContext({ model });
  const session = new LlamaChatSession({ context });

  return session;
};

export { Session };