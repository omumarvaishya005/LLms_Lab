from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='Book',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# docs = loader.load() # slow and do eager loading and consume all memory and time 

docs = loader.lazy_load()
print(type(docs))
for document in docs: 
    print(document.metadata)
docs=list(docs)
print(len(docs))
