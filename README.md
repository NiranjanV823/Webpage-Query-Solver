I implemented a Retrieve-and-Generate (RAG) approach using the Starling-7B large language model from the Hugging Face library.  
I utilized VectorStoreIndex for data chunking and storage, enabling me to efficiently retrieve the top three relevant chunks using a query engine.  
To enhance performance, I employed 4-bit vector quantization with BitsAndBytesConfig, significantly reducing memory usage while maintaining the model's effectiveness.
