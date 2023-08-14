---
layout: post
title: Incorporating AI into Applications - A Survey
date: 2023-05-12 15:53:00-0400
description: An overview of the current state of AI in applications, and how we can move from Software 2.0 to 3.0
categories: AI LLMs
giscus_comments: true
related_posts: true
---

There already exists a wealth of information on how LLMs can be used to accelerate productivity in the process of
application design, development, and testing.

This post provides a survey of new tools and techniques for incorporating user-facing AI-driven functionality into
applications.

---

## Semantic Search and Embeddings

Semantic search is a powerful information-retrieval technique that significantly enhances search functionality. It is
also one of the building blocks needed to give LLM-powered chatbots long-term memory as well as the ability for chatbots
to answer questions about text data sets given a natural language query (via retrieval augmented generation).

Searching for text data in documents or databases has traditionally been done with some variant of lexical/keyword/fuzzy
search (e.g. ElasticSearch, Solr, PostgreSQL FTS).

A semantic search aims not only to interpret search keywords, but also strives to determine the intent and contextual
meaning behind a search query. For example, if you have a database of e-commerce product reviews, a semantic search
would return relevant results given a query like “Find me the negative reviews of product X”, even if there was no
prior labeling of the reviews as positive/negative.

Embeddings are at the core of semantic search, and embedding models like OpenAI’s [text-embedding-ada-002](https://openai.com/blog/new-and-improved-embedding-model) are cheap,
powerful, and easier to integrate into applications than ever before.

Embeddings are representations of data as points/vectors in n-dimensional space, where data points that are similar in
meaning cluster together.

A basic semantic search that finds the most relevant pieces of data for a user’s query can be implemented by:

1. Generating and storing embeddings for the data to be searched
2. Generating an embedding of the user’s query
3. Finding the k-nearest-neighbor (kNN) embeddings to the query embedding.

To see what a semantic search via cosine similarity looks like in code, here's a very basic example demonstrating
exact nearest neighbor (ENN) search to find the top k most relevant records for a query.

```typescript
const computeDotProduct = (a: number[], b: number[]) =>
    a.map((x, i) => a[i] * b[i]).reduce((m, n) => m + n);

const getTopKNearestNeighbors = async (recordsToSearch: Record[], searchQuery: string, k: number) => {
    const embeddedQuery = await getEmbeddingForString(searchQuery);

    // Create a min-heap to keep track of of the top k most relevant results
    const heap = new Heap<[number, Record]>((a, b) => a[0] - b[0]);
    for (const record of recordsToSearch) {
        const recordEmbedding = record.getEmbedding();
        const dotProduct = this.computeDotProduct(embeddedQuery, record.getEmbedding());

        if (heap.size() < k) {
            heap.push([dotProduct, record]);
        } else if (dotProduct > heap.peek()![0]) {
            heap.pop();
            heap.push([dotProduct, record]);
        }
    }

    return Array.from(heap.toArray())
        .sort((a, b) => b[0] - a[0])
        .map(recordWithDotProduct => recordWithDotProduct[1]);
};
```

Embeddings can be generated for long unstructured data via chunking (e.g. textbooks, wikipedia articles) as well as
structured data (rows in a database). They can be stored in specialized vector databases, or even [relational](https://github.com/pgvector/pgvector) databases.

There are multiple techniques for enhancing semantic search:

- Hypothetical Document Embeddings ([HyDE](https://wfhbrian.com/revolutionizing-search-how-hypothetical-document-embeddings-hyde-can-save-time-and-increase-productivity/))
- Using a [rerank](https://txt.cohere.com/rerank/) model to optimize search result ordering
- Using a [hybrid Lexical + Semantic search](https://js.langchain.com/docs/modules/indexes/retrievers/supabase-hybrid), combined with a rerank.
- Pre/Post filtering using metadata

Embeddings have many other applications beyond semantic search, like recommendations, clustering, classification, and
anomaly detection, to name a few.

---

## Chatbots Armed with Data

The classic first-order use case for LLMs in applications is a chatbot that can respond to queries about
specific/private data without hallucinating. For example, if you have a private database of company policies,
a question you could ask a chatbot with access to this data is “What is my company's PTO policy?”.

The general approach is to find all data relevant to the user’s query (e.g. with a semantic search), then ‘stuff’ as
much of the relevant data as possible into an LLM prompt along with the user’s query, and ideally get a data-backed
response in natural language. This is how the ChatGPT retrieval plugin works. Various techniques exist to
enhance/augment this process as well.

Especially when combined with voice transcription - chatbots with semantic search capabilities can be a powerful feature
for users that have questions about text-heavy data sets.

“Long term memory” for chatbots can also be approximated by making all previous conversations semantically searchable
and “recalling” the most relevant ones for the current conversation via semantic search.

However, stuffing semantic search results is not a panacea. A user cannot conduct statistical/numerical analysis over a
data set with this approach, and users may not want natural language as the input/output.

---

## Beyond Chatbots: Semi-Autonomous LLM Agents with Tools

LLMs are game-changing when we augment them with tools and memory, and use them as reasoning engines within our code, as
seen with ChatGPT plugins.

If given access to an environment where they can execute code, LLMs can write SQL scripts or Python code to answer
complex statistical/numerical analysis queries. The ChatGPT code interpreter plugin is a prime example.

If given access to a file system, LLMs can accept inputs and generate outputs other than natural language (e.g. charts,
PDF reports, executable files).

If given access to a ‘memory’ datastore (typically a vector DB), previously generated code and conversations can be
stored and retrieved (typically via a semantic search). LLMs can even progressively build up a [skill](https://voyager.minedojo.org/assets/documents/voyager.pdf) library where
generated functions are composed together to achieve tasks of ever-increasing complexity.

If given access to a human, agents can be provided with additional guidance or clarification in order to perform its
task correctly.

LLMs can be used as reasoning engines within applications to decide which tool or branch of logic is appropriate to
satisfy a user’s query.

## A Powerful Example

Here’s a concrete example of a (hypothetical) feature that can be built today with current technology:

1. User voice query: “Generate a weekly PDF report containing XYZ statistical data about our customers and email it to
me@example.com”
2. Input transcribed to text and provided to “plan-and-execute” LLM.
3. LLM generates SQL+python code to conduct data analysis/visualization to satisfy the user’s query. Code iterated upon (
autonomously), validated, and then executed.
4. LLM generates code to save data analysis and visualization output into a PDF and programmatically upload the report to
Amazon S3, and send an email with a link to the PDF..
5. LLM saves the generated code into long term memory in a vector database for future reuse in similar or more complex
tasks.
6. LLM uses the AWS SDK to programmatically create an AWS Lambda function that generates and emails an updated report every
week using the generated code.

## Other Considerations 

- Prompt Injection is a major issue for applications that integrate user-facing LLMs.
- Reliability of LLM outputs is another hard problem. Frameworks are emerging to address this.
- LLM latency, context windows, and specialization to perform certain tasks will all improve over time.
- Evaluation of LLM agent performance will become an important QA discipline.
- Langchain seems to be the center for cutting-edge developments with respect to incorporating LLMs into user facing
applications.
- As AI models get smaller and faster, they will be able to run locally using things like WebGPU, mitigating privacy
concerns.