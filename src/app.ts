import "dotenv/config";
import "pdf-parse";

import { HuggingFaceInference } from "@langchain/community/llms/hf";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { formatDocumentsAsString } from "langchain/util/document";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";

(async () => {
  const llm = new HuggingFaceInference({
    model: "meta-llama/Meta-Llama-3-8B-Instruct",
    // model: "mistralai/Mistral-7B-Instruct-v0.2",
    // model: "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
  });
  const embeddings = new HuggingFaceInferenceEmbeddings();

  const loader = new PDFLoader("document.pdf");

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
  // custom RAG prompt
  const template = `Use the following pieces of context to answer the question at the end.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.
  Use three sentences maximum and keep the answer as concise as possible.
  Always say "thanks for asking!" at the end of the answer.

  {context}

  Question: {question}

  Helpful Answer:`;
  const prompt = ChatPromptTemplate.fromTemplate(template)

  const declarativeRagChain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocumentsAsString),
      question: new RunnablePassthrough(),
    },
    prompt,
    llm,
    new StringOutputParser(),
  ]);

  console.log("Asking question...");
  const res = await declarativeRagChain.invoke("What is the content of the context pieces about?");
  console.log(res);
})();