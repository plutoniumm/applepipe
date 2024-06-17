import { Session } from "./model";
import { Hono } from 'hono';

const PORT = process.env.PORT || 3000;
const session = Session("gemma-2b-q8.gguf");

const modes = {
  text: (q) => q,
  ques: q => `Q: ${q}\n\nA: `
};

const app = new Hono();
app.get('/predict', async (c) => {
  const mode = c.req.query('mode') || "text";
  let query = "";
  try {
    query = await c.req.text();
  } catch (e) {
    query = c.req.query('q') || "";
  };
  if (query.length < 1) return;

  const Query = modes[mode](query);
  const Answer = await session.prompt(Query, {
    temperature: 0.05,
    maxTokens: 5
  });

  return c.text(Answer);
});

export default {
  port: PORT,
  fetch: app.fetch,
}