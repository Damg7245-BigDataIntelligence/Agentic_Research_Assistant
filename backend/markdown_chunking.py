import re
import json

def chunk_markdown_by_headers(markdown_text, ideal_word_count=500):
    """
    Splits the markdown content into chunks based on headers.
    If a chunk's word count is greater than 1.5 * ideal_word_count,
    splits that chunk into two parts with the same header.
    
    Each chunk is represented as a dictionary containing:
        - 'header': The header line (if available)
        - 'level': The header level (number of '#' characters)
        - 'content': The text content for that chunk
        - 'part' (optional): For split chunks, indicates part 1 or 2.
    
    Parameters:
        markdown_text (str): The full markdown text to be chunked.
        ideal_word_count (int): Ideal word count for a chunk (default is 500).
    
    Returns:
        List[Dict]: A list of chunk dictionaries with header metadata.
    """
    # Define the threshold to split: 1.5 * ideal_word_count
    split_threshold = int(1.5 * ideal_word_count)
    
    # Regex to match markdown headers at the beginning of a line.
    header_pattern = re.compile(r'^(#{1,6})\s*(.+)$', re.MULTILINE)
    
    # Find all header matches with their start positions.
    matches = list(header_pattern.finditer(markdown_text))
    chunks = []
    
    # If no header is found, treat the entire text as one chunk.
    if not matches:
        chunks.append({
            'header': None,
            'level': None,
            'content': markdown_text.strip()
        })
        return chunks
    
    # Iterate through headers and split text between them.
    for idx, match in enumerate(matches):
        header_line = match.group(0).strip()  # full header line
        header_level = len(match.group(1))      # number of '#' determines level
        start_index = match.start()
        end_index = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown_text)
        chunk_text = markdown_text[start_index:end_index].strip()
        
        # Split the chunk into words to check length.
        words = chunk_text.split()
        if len(words) > split_threshold:
            # Split into two parts (roughly equal halves).
            mid_point = len(words) // 2
            part1_text = " ".join(words[:mid_point])
            part2_text = " ".join(words[mid_point:])
            
            chunks.append({
                'header': header_line,
                'level': header_level,
                'content': part1_text,
                'part': 1
            })
            chunks.append({
                'header': header_line,
                'level': header_level,
                'content': part2_text,
                'part': 2
            })
        else:
            chunks.append({
                'header': header_line,
                'level': header_level,
                'content': chunk_text
            })
    
    return chunks

# # --- Main Section ---
# if __name__ == "__main__":
#     file_path = "/Users/janvichitroda/Documents/Janvi/NEU/Big_Data_Intelligence_Analytics/Assignment 5/Part 1/Github_Repo/Agentic_Research_Assistant/input/2022_Fourth_Quarter.md"
    
#     # Read the file content
#     with open(file_path, "r", encoding="utf-8") as f:
#         sample_markdown = f.read()
    
#     # Get chunks from the markdown content
#     chunks = chunk_markdown_by_headers(sample_markdown)

#     output_file = "/Users/janvichitroda/Documents/Janvi/NEU/Big_Data_Intelligence_Analytics/Assignment 5/Part 1/Github_Repo/Agentic_Research_Assistant/input/output_chunks.json"

#     # Write the chunks list to the JSON file
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(chunks, f, ensure_ascii=False, indent=4)

#     print(f"Chunks successfully saved to {output_file}")