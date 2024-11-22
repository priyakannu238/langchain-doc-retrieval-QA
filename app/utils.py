import tiktoken
import pinecone

def calculate_token_cost(texts):
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    cost = total_tokens / 1000 * 0.0004
    return total_tokens, cost


def print_embedding_cost(texts):
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    Embedding_Cost_in_USD = f"{total_tokens / 1000 * 0.00002:.6f}"
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {Embedding_Cost_in_USD}')

    return total_tokens, Embedding_Cost_in_USD



def sanitize_index_name(name):
    """
    Sanitize the index name to meet Pinecone's requirements:
    - Only lowercase alphanumeric characters and hyphens
    - Maximum length of 45 characters
    - Cannot start or end with a hyphen
    """

    sanitized = name.lower()
    sanitized = ''.join(c if c.isalnum() else '-' for c in sanitized)

    while '--' in sanitized:
        sanitized = sanitized.replace('--', '-')

    sanitized = sanitized[:30]
    sanitized = sanitized.strip('-')
    
    if not sanitized:
        sanitized = 'document-index'

    return sanitized


def delete_pinecone_index():
    pc = pinecone.Pinecone()

    index_info = pc.list_indexes()

    if index_info:
        indexes = [index['name'] for index in index_info]
        print(f"Deleting the following indexes: {', '.join(indexes)} ...")
        for index_name in indexes:
            pc.delete_index(index_name)
        print("All indexes deleted successfully.")
    else:
        print("No indexes to delete.")
