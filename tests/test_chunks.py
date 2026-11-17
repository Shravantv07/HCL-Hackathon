from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import chunk_pages, chunk_figures
from src.ingestion.figure_extractor import extract_figures

pdf = 'data/papers/2509.23661v3.pdf'
pages = parse_pdf(pdf, 'p01')
print(f'Pages:      {len(pages)}')

chunks = chunk_pages(pages, paper_title='LLaVA-OneVision-1.5')
print(f'Chunks:     {len(chunks)}')
print(f'Text:       {sum(1 for c in chunks if c["chunk_type"]=="text")}')
print(f'Table:      {sum(1 for c in chunks if c["chunk_type"]=="table")}')

figs = extract_figures(pdf, 'p01', 'data/figures')
print(f'Figures:    {len(figs)}')

fig_chunks = chunk_figures(figs, paper_title='LLaVA-OneVision-1.5')
print(f'Fig chunks: {len(fig_chunks)}')

print()
print('--- Sample text chunk ---')
print(f'Page:    {chunks[0]["page"]}')
print(f'Section: {chunks[0]["section"]}')
print(f'Tokens:  {chunks[0]["token_count"]}')
print(f'Text:    {chunks[0]["text"][:300]}')

if figs:
    print()
    print('--- Sample figure ---')
    print(f'ID:      {figs[0]["fig_id"]}')
    print(f'Caption: {figs[0]["caption"][:100]}')
