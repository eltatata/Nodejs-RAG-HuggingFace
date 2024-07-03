import "dotenv/config";

import { HuggingFaceInference } from "@langchain/community/llms/hf";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { formatDocumentsAsString } from "langchain/util/document";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";

(async () => {
  const llm = new HuggingFaceInference({
    // model: "meta-llama/Meta-Llama-3-8B-Instruct",
    model: "mistralai/Mistral-7B-Instruct-v0.2",
    // model: "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
  });
  const embeddings = new HuggingFaceInferenceEmbeddings();

  const loader = new CheerioWebBaseLoader(
    "https://js.langchain.com/v0.2/docs/tutorials/rag"
  );

  console.log("Loading documents...");
  const docs = await loader.load();

  console.log("Splitting documents...");
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const splits = await textSplitter.splitDocuments(docs);

  console.log("Generating embeddings...");
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splits,
    embeddings
  );

  // Retrieve and generate using the relevant snippets of the blog.
  const retriever = vectorStore.asRetriever();

  const contextualizeQSystemPrompt = `Given a chat history and the latest user question
  which might reference context in the chat history, formulate a standalone question
  which can be understood without the chat history. Do NOT answer the question,
  just reformulate it if needed and otherwise return it as is.`;

  const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    ["system", contextualizeQSystemPrompt],
    new MessagesPlaceholder("chat_history"),
    ["human", "{question}"],
  ]);
  const contextualizeQChain = contextualizeQPrompt
    .pipe(llm)
    .pipe(new StringOutputParser());


  const qaSystemPrompt = `You are an assistant for question-answering tasks.
  Use the following pieces of retrieved context to answer the question.
  If you don't know the answer, just say that you don't know.
  Use three sentences maximum and keep the answer concise.

  {context}`;

  const qaPrompt = ChatPromptTemplate.fromMessages([
    ["system", qaSystemPrompt],
    new MessagesPlaceholder("chat_history"),
    ["human", "{question}"],
  ]);

  const contextualizedQuestion = (input: Record<string, unknown>) => {
    if ("chat_history" in input) {
      return contextualizeQChain;
    }
    return input.question;
  };

  const ragChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      context: (input: Record<string, unknown>) => {
        if ("chat_history" in input) {
          const chain: any = contextualizedQuestion(input);
          return chain.pipe(retriever).pipe(formatDocumentsAsString);
        }
        return "";
      },
    }),
    qaPrompt,
    llm,
  ]);

  let chat_history: string[]  = [];

  const firstQuestion = "What is a RAG?";
  const response1 = await ragChain.invoke({ question: firstQuestion, chat_history });
  console.log(response1);
  chat_history = chat_history.concat(response1);

  const secondQuestion = "How can i make one?";
  const response2 = await ragChain.invoke({ question: secondQuestion, chat_history });
  console.log(response2);
  chat_history = chat_history.concat(response2);

  const thirdQuestion = "And what concepts should you have to build one?";
  const response3 = await ragChain.invoke({ question: thirdQuestion, chat_history });
  console.log(response3);
})();