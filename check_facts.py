import chromadb

client = chromadb.PersistentClient(path='facts_db')
collection = client.get_collection('facts')

print(f"Total facts: {collection.count()}")
print()

print("=" * 50)
print("BIRTH FACTS")
print("=" * 50)
results = collection.query(query_texts=['birth born birthplace'], n_results=5)
for f, m in zip(results['documents'][0], results['metadatas'][0]):
    print(f"  [{m['book']} p{m['page']}] {f[:150]}")

print()
print("=" * 50)
print("SATYAGRAHA FACTS")
print("=" * 50)
results = collection.query(query_texts=['satyagraha meaning definition'], n_results=5)
for f, m in zip(results['documents'][0], results['metadatas'][0]):
    print(f"  [{m['book']} p{m['page']}] {f[:150]}")

print()
print("=" * 50)
print("GANDHI FACTS")
print("=" * 50)
results = collection.query(query_texts=['who was gandhi mahatma'], n_results=5)
for f, m in zip(results['documents'][0], results['metadatas'][0]):
    print(f"  [{m['book']} p{m['page']}] {f[:150]}")